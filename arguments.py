import os, argparse
import itertools


def get_args():
    """
    Returns
        system_args (dict): path, log setting
        experiment_args (dict): hyper-parameters
        args (dict): system_args + experiment_args
    """
    system_args = {
        # expeirment info
        'name'          : 'MCL-SV',
        'project'       : 'MCL-SV',
        'tags'          : ['01'],

        'wandb_mode'    : False,
        'wandb_entity'  : '',
        'wandb_api_key' : '',
        'wandb_group'   : '',
        'description'   : '',

        ## Load and save
        'initial_model' : 'pretrained_weights/Exp7.DDSE_ReDimNet(SV+MC)_Vox2.pt', 
        'save_path'     : '/exps',

        ## Training and test data
        'path_vox1'     : '/data/VoxCeleb1',
        'path_vox2'     : '/data/VoxCeleb2',
        'path_musan'    : '/data/musan',
        'path_musan_split': '/data/musan_train_test_split',
        'path_rir'      : '/data/RIRS_NOISES/simulated_rirs',
        'path_voxsrc23' : '/data/VoxSRC2023_dev',
        'path_vcmix'    : '/data/wo_ovl',
        'path_nonspeech': '/data/nonspeech',
    }

    experiment_args = {
        'model' : 'DDSE_ReDimNet',
        'C': 16,
        'F': 72,
        'embed_dim'     : 192,
        'global_context_att': True,
        'group_divisor' : 4,
        'hop_length'    : 160,
        'out_channels'  : None,
        'pooling_func'  : 'ASTP',
        'stages_setup': [[1, 2, 12], 
                [2, 2, 12],
                [1, 3, 12],
                [3, 4, 8],
                [1, 4, 8],
                [2, 4, 4]],

        ## Data loader
        'frame_size'        : 199,
        'batch_size'        : 50,
        'eval_frames'       : 0,
        'augment'           : True,
        'max_epoch'         : 50,
        'seed'              : 10,
        'n_loader_thread'   : 5,
        'test_interval'     : 1,
        'max_seg_per_spk'   : 500,
        'n_classes'         : 5994,

        ## Loss functions
        'trainfunc'         : 'sphereface2', 
        'margin'            : 0.2,
        'scale'             : 32.0,
        'nPerSpeaker'       : 1,
        'hard_prob'         : 0.5,
        'hard_rank'         : 10,

        ## SE Loss functions
        'trainfunc_se'      : 'mse',

        ## Optimizer and Scheduler 
        'optimizer'         : 'adam',
        'scheduler'         : 'cosine_warmup',
        'lr'                : 1e-6,
        'T_0'               : 20,
        'T_mult'            : 1,
        'eta_max'           : 1e-3,
        'gamma'             : 0.5,
        'weight_decay'      : 2e-5,
        'lr_decay'          : 0.95,

        ## Evaluation parameters
        'dcf_p_target'      : 0.05,
        'dcf_c_miss'        : 1,
        'dcf_c_fa'          : 1,

    }


    parser = argparse.ArgumentParser()
    for key, value in itertools.chain(system_args.items(), experiment_args.items()):
        parser.add_argument(f"--{key}", type=type(value), default=value)
        #print(f"--{key}, type={type(value)}, default={value}")
    parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
    parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
    
    args = parser.parse_args()
    args.path_scripts = os.path.dirname(os.path.realpath(__file__))

    return args
