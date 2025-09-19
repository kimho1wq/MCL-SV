#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib

from data.loader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, trainfunc, trainfunc_se, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        LossFunction_SE = importlib.import_module("loss." + trainfunc_se).__getattribute__("LossFunction")
        self.__L_SE__ = LossFunction_SE(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        emb, spec = self.__S__.forward(data, label)

        if label != None:
            with torch.no_grad():
                data_spec = self.__S__.stft(data)
            nloss_mc = self.__L_SE__.forward(spec['se']+spec['ne'], data_spec)

            emb = emb.reshape(self.nPerSpeaker, -1, emb.size()[-1]).transpose(1, 0).squeeze(1)

            nloss_spk, prec1 = self.__L__.forward(emb, label)

            return nloss_spk, nloss_mc, prec1
        
        else:
            return emb

            


class ModelTrainer(object):
    def __init__(self, speaker_model, logger, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        self.logger = logger

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose, freeze=True):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss_spk = 0
        loss_mc = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        for step, (data, data_label) in enumerate(loader):
            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).cuda()

            nloss_spk, nloss_mc, prec1 = self.__model__(data, label)
            nloss = nloss_spk + nloss_mc
            nloss.backward()
            self.__optimizer__.step()

            loss_spk += nloss_spk.detach().cpu().item()
            loss_mc += nloss_mc.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                clr = [x['lr'] for x in self.__optimizer__.param_groups]
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write(" SPK {:f} MC {:f} - {:.2f} Hz, LR: {:.6f}".format(loss_spk / counter, loss_mc / counter, stepsize / telapsed, max(clr)))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss_spk / counter, loss_mc / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_dataset, trials, n_loader_thread, distributed, short_size=None, ref_feats=None, print_interval=100, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()


        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=n_loader_thread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        #te = []
        for idx, data in enumerate(test_loader):
            with torch.no_grad():
                inp1 = data[0][0].cuda()
                ref_feat = self.__model__(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart
            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, trial in enumerate(trials):
                key1 = trial.key1
                key2 = trial.key2
                label = trial.label
                #data = line.split()
                ## Append random label if missing
                #if len(data) == 2:
                #    data = [random.randint(0, 1)] + data
                
                ref_feat = ref_feats[key1].cuda() if ref_feats else feats[key1].cuda()
                com_feat = feats[key2].cuda()
                #ref_feat = ref_feats[data[1]].cuda() if ref_feats else feats[data[1]].cuda()
                #com_feat = feats[data[2]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(1, -1), com_feat.reshape(1, -1)).detach().cpu().numpy()
                
                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(label))
                all_trials.append(key1 + " " + key2)

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials, feats)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)


