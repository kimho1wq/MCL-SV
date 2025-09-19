#! /usr/bin/python
# -*- encoding: utf-8 -*-

import os
import random
import torch
import numpy
import os
import glob
import soundfile
from scipy import signal 
from scipy.io import wavfile
from torch.utils.data import Dataset
import torch.distributed as dist
from .musan import MusanNoise

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(filename, frame_size, short_size=None, evalmode=True, num_eval=10):
    
    # Maximum audio length
    division_len = 160 * 8
    frame_size = frame_size * 160 
    
    # Read wav file and convert to torch tensor
    audio, sr = soundfile.read(filename)

    audiosize = audio.shape[0]
    if audiosize <= frame_size:
        shortage    = frame_size - audiosize + 1 
        audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]
    elif audiosize > int(70*sr):
        audio = audio[:int(70*sr)]
        audiosize   = audio.shape[0]
 
    feats = []
    if evalmode:
        if frame_size == 0:
            feats.append(audio if (audiosize + 160) % division_len == 0 
                         else audio[:-(audiosize % division_len + 160)])
        else:
            startframe = numpy.linspace(0,audiosize-frame_size,num=num_eval)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+frame_size])
    else:
        #startframe = numpy.linspace(0,audiosize-frame_size,num=10)
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-frame_size))])
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+frame_size])

    feat = numpy.stack(feats,axis=0).astype(numpy.float64)
    
    return feat;



class AugmentWAV(object):

    def __init__(self, path_musan, path_musan_split, path_rir):

        self.musan = MusanNoise(f'{path_musan_split}/train')
        self.rir_files  = glob.glob(os.path.join(path_rir,'*/*/*.wav'));

    def additive_noise(self, audio, category=None):
        return self.musan(audio, category) # noise injection

    def reverberate(self, audio, frame_size):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float64),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:frame_size * 160]


class train_dataset_loader(Dataset):
    def __init__(self, train_db, augment, path_musan, path_musan_split, path_rir, frame_size, **kwargs):

        self.augment_wav = AugmentWAV(path_musan, path_musan_split, path_rir)

        #self.train_list = train_list
        self.frame_size = frame_size;
        self.augment    = augment
        self.p = 0.5

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for idx, item in enumerate(train_db):
            self.data_list.append(item.path)
            self.data_label.append(item.label)
      

    def __getitem__(self, indices):
     
        audio_l = []

        for index in indices:
            frame_size = self.frame_size

            audio = loadWAV(self.data_list[index], frame_size, evalmode=False)

            augtype = random.randint(0,5)
            if augtype == 1:
                audio   = self.augment_wav.reverberate(audio, frame_size)
            elif augtype == 2:
                audio, _   = self.augment_wav.additive_noise(audio, 'music')
            elif augtype == 3:
                audio, _ = self.augment_wav.additive_noise(audio, 'speech') # noise injection
            elif augtype == 4:
                audio, _ = self.augment_wav.additive_noise(audio, 'noise') # noise injection
            elif augtype == 5: #Television noise
                audio, _ = self.augment_wav.additive_noise(audio, 'speech') # noise injection
                audio, _ = self.augment_wav.additive_noise(audio, 'music') # noise injection
                
            audio_l.append(audio)

        audio_l = numpy.concatenate(audio_l, axis=0)

        return torch.FloatTensor(audio_l), self.data_label[index]

    def __len__(self):
        return len(self.data_list)




class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label         = data_source.data_label;
        self.nPerSpeaker        = nPerSpeaker;
        self.max_seg_per_spk    = max_seg_per_spk;
        self.batch_size         = batch_size;
        self.epoch              = 0;
        self.seed               = seed;
        self.distributed        = distributed;
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = [];
            data_dict[speaker_label].append(index);


        ## Group file indices for each class
        dictkeys = list(data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []
        
        for findex, key in enumerate(dictkeys):
            data    = data_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            
            rp      = lol(numpy.arange(numSeg),self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
            start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
            end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch



class test_dataset_loader(Dataset):
    def __init__(self, db_list, eval_frames, short_size=None, **kwargs):
        self.eval_frames = eval_frames
        self.db_list  = db_list
        self.short_size  = short_size
    def __getitem__(self, index):
        
        audio = loadWAV(self.db_list[index].path, self.eval_frames, short_size=self.short_size, evalmode=True)
        return torch.FloatTensor(audio), self.db_list[index].key

    def __len__(self):
        return len(self.db_list)

