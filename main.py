#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse 
import yaml
import torch
import glob
import zipfile
import warnings
import datetime
from utils import *
from train import *
from data.loader import *
import torch.distributed as dist
import torch.multiprocessing as mp
from log.controller import LogModuleController
from data.preprocessing import DataPreprocessor
import arguments
from data.voxsrc import VoxSRC23
from data.vox1_noise import VoxCeleb1_Noise
from data.vox2 import VoxCeleb2
from data.vcmix import VCMix
from data.nonspeech import NonSpeech


warnings.simplefilter("ignore")



## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    print(f"\nmain_worker gpu: {gpu}")

    args.gpu = gpu
    ## Load models
    s = SpeakerNet(**vars(args))
    print(f"__S__ model n_params: {sum(p.numel() for p in s.__S__.parameters() if p.requires_grad)}")

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)


    if args.gpu == 0:
        ## Write args to scorefile
        args.result_save_path    = os.path.join(args.save_path, args.name, "metric")
        os.makedirs(args.result_save_path, exist_ok=True)
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    ## Initialise trainer and data loader
    db_vox2 = VoxCeleb2(args.path_vox2)
    db_vox1= VoxCeleb1_Noise(args.path_vox1)
    db_voxsrc23 = VoxSRC23(args.path_voxsrc23)
    db_vcmix = VCMix(args.path_vcmix, args.path_vox1)
    db_nonspeech = NonSpeech(args.path_nonspeech)


    test_dbs = {
        'Vox1': (db_vox1.test_items, db_vox1.trials, None),
        'Vox1_Noise_0': (db_vox1.test_items_noise['noise_0'], db_vox1.trials, 'Vox1'),
        'Vox1_Noise_5': (db_vox1.test_items_noise['noise_5'], db_vox1.trials, 'Vox1'),
        'Vox1_Noise_10': (db_vox1.test_items_noise['noise_10'], db_vox1.trials, 'Vox1'),
        'Vox1_Noise_15': (db_vox1.test_items_noise['noise_15'], db_vox1.trials, 'Vox1'),
        'Vox1_Noise_20': (db_vox1.test_items_noise['noise_20'], db_vox1.trials, 'Vox1'),
        'Vox1_Music_0': (db_vox1.test_items_noise['music_0'], db_vox1.trials, 'Vox1'),
        'Vox1_Music_5': (db_vox1.test_items_noise['music_5'], db_vox1.trials, 'Vox1'),
        'Vox1_Music_10': (db_vox1.test_items_noise['music_10'], db_vox1.trials, 'Vox1'),
        'Vox1_Music_15': (db_vox1.test_items_noise['music_15'], db_vox1.trials, 'Vox1'),
        'Vox1_Music_20': (db_vox1.test_items_noise['music_20'], db_vox1.trials, 'Vox1'),
        'Vox1_Speech_0': (db_vox1.test_items_noise['speech_0'], db_vox1.trials, 'Vox1'),
        'Vox1_Speech_5': (db_vox1.test_items_noise['speech_5'], db_vox1.trials, 'Vox1'),
        'Vox1_Speech_10': (db_vox1.test_items_noise['speech_10'], db_vox1.trials, 'Vox1'),
        'Vox1_Speech_15': (db_vox1.test_items_noise['speech_15'], db_vox1.trials, 'Vox1'),
        'Vox1_Speech_20': (db_vox1.test_items_noise['speech_20'], db_vox1.trials, 'Vox1'),
        'Nonspeech_0': (db_nonspeech.items[0], db_vox1.trials, 'Vox1'),
        'Nonspeech_5': (db_nonspeech.items[5], db_vox1.trials, 'Vox1'),
        'Nonspeech_10': (db_nonspeech.items[10], db_vox1.trials, 'Vox1'),
        'Nonspeech_15': (db_nonspeech.items[15], db_vox1.trials, 'Vox1'),
        'Nonspeech_20': (db_nonspeech.items[20], db_vox1.trials, 'Vox1'),
        'VCMix': (db_vcmix.items, db_vcmix.trials, None),
        'VoxSRC23': (db_voxsrc23.items, db_voxsrc23.trials, None),
    }
    

    train_dataset = train_dataset_loader(db_vox2.train_items, **vars(args))
    train_sampler = train_dataset_sampler(train_dataset, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_loader_thread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    args.number_iteration = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"args.number_iteration: {args.number_iteration}")

    if args.gpu == 0:
        builder = LogModuleController.Builder(args.name, args.project
            ).tags(args.tags
            ).save_source_files(args.path_scripts
            ).use_local(args.save_path)
        if args.wandb_mode:
            builder.use_wandb(args.wandb_group, args.wandb_entity, args.wandb_api_key)
            
        logger = builder.build()
        logger.log_parameter(vars(args))
    else:
        logger = None

    

    eers = {}
    for k in test_dbs.keys():
        eers[k] = [100]
    ref_feats = {None: None}

    trainer     = ModelTrainer(s, logger, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    it = 1
    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1


    if args.eval:
        it = 0
        for key, value in test_dbs.items():
            sc, lab, _, feats = trainer.evaluateFromList(
                test_dataset=test_dataset_loader(value[0], **vars(args)),
                trials=value[1], ref_feats=ref_feats[value[2]], **vars(args))
            if key == 'Vox1':
                ref_feats[key] = feats

            if args.gpu == 0:
                eers = cal_results(args, eers, key, sc, lab, it, scorefile, logger)
        exit(0)



    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Core training script
    for it in range(it,args.max_epoch+1):
        
        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss_spk, loss_sum, _ = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, SPK {:f}, SUM {:f}, LR {:f}".format(it, loss_spk, loss_sum, max(clr)))
            scorefile.write("Epoch {:d}, SPK {:f}, SUM {:f}, LR {:f} \n".format(it, loss_spk, loss_sum, max(clr)))
            logger.log_metric("spk_loss", loss_spk, epoch_step = it)
            logger.log_metric("sum_loss", loss_sum, epoch_step = it)
            logger.log_metric("lr", max(clr), epoch_step = it)
        

        for key, value in test_dbs.items():
            sc, lab, _, feats = trainer.evaluateFromList(
                test_dataset=test_dataset_loader(value[0], **vars(args)),
                trials=value[1], ref_feats=ref_feats[value[2]], **vars(args))
            if key == 'Vox1':
                ref_feats[key] = feats

            if args.gpu == 0:
                eers = cal_results(args, eers, key, sc, lab, it, scorefile, logger)
        

        if args.gpu == 0:
            eer_list = [value[-1] for key, value in eers.items() if 'Vox1' in key]
            if len(eer_list) != 0:
                logger.log_metric("AVG_EER_Vox1", sum(eer_list)/len(eer_list), epoch_step = it)
                       
            eer_list = [value[-1] for key, value in eers.items() if 'Nonspeech' in key]
            if len(eer_list) != 0:
                logger.log_metric("AVG_EER_Nonspeech", sum(eer_list)/len(eer_list), epoch_step = it)

            trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)
            

    if args.gpu == 0:
        scorefile.close()
        logger.finish()


def cal_results(args, eers, key, sc, lab, it, scorefile, logger):

    result = tuneThresholdfromScore(sc, lab, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

    print(f'\n=== {key} ===\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
    scorefile.write(f"   {key}   ")
    scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

    if '_' in key:
        parts = key.split('_')
        if len(parts) == 3:
            test_name = parts[1]
            snr = parts[-1]
        else:
            test_name = key
            snr = None
    else:
        test_name = key
        snr = None

    eers[key].append(result[1])
    if snr is not None:
        logger.log_metric(f"{test_name}/EER_{snr}", result[1], epoch_step = it)
        logger.log_metric(f"{test_name}/EER_BEST_{snr}", min(eers[key]), epoch_step = it)
        logger.log_metric(f"{test_name}/MinDCF_{snr}", mindcf, epoch_step = it)
    else:
        logger.log_metric(f"{test_name}/EER", result[1], epoch_step = it)
        logger.log_metric(f"{test_name}/EER_BEST", min(eers[key]), epoch_step = it)
        logger.log_metric(f"{test_name}/MinDCF", mindcf, epoch_step = it)
    

    with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
        eerfile.write('{:2.4f}'.format(result[1]))

    scorefile.flush()

    return eers


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args = arguments.get_args()
    
    args.port = f'10{datetime.datetime.now().microsecond % 100}'
    args.model_save_path = os.path.join(args.save_path, args.name, "model")
    os.makedirs(args.model_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Port:', args.port)
    print('Save path:', args.save_path)
    print("args.distributed:", args.distributed)

    # check musan dataset
    data_preprocessor = DataPreprocessor(args.path_musan, args.path_vox1)
    data_preprocessor.check_environment()


    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()