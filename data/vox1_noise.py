import os
from dataclasses import dataclass

NUM_TRAIN_ITEM = 148642
NUM_TRAIN_SPK = 1211
NUM_TRIALS = 37611
NUM_TRIALS_E = 579818
NUM_TRIALS_H = 550894
NUM_TEST_ITEM = 148642+4874

@dataclass
class TrainItem:
    path: str
    speaker: str
    label: int

@dataclass
class EnrollmentItem:
    key: str
    path: str

@dataclass
class Trial:
    key1: str
    key2: str
    label: int

class VoxCeleb1_Noise:
    def __init__(self, path_vox1):
        self.train_items = []
        self.test_items = []
        self.test_items_E = []

        self.trials = []
        self.trials_H = []
        self.trials_E = []
        self.class_weight = []

        noises = ['noise_0','noise_5','noise_10','noise_15','noise_20',
                  'speech_0','speech_5','speech_10','speech_15','speech_20',
                  'music_0','music_5','music_10','music_15','music_20',
                  ]
        self.test_items_noise = { i : [] for i in noises}

        # train_set
        labels = {}
        num_utt = [0 for _ in range(NUM_TRAIN_SPK)]
        num_sample = 0
        for root, _, files in os.walk(os.path.join(path_vox1, 'train')):
            for file in files:
                if '.wav' in file:
                    # combine path
                    f = os.path.join(root, file)
                    
                    # parse speaker
                    spk = f.split('/')[-3]
                    
                    # labeling
                    try: labels[spk]
                    except: 
                        labels[spk] = len(labels.keys())

                    # init item
                    item = TrainItem(path=f, speaker=spk, label=labels[spk])
                    self.train_items.append(item)
                    num_sample += 1
                    num_utt[labels[spk]] += 1

        for n in num_utt:
            self.class_weight.append(num_sample / n)
                    
        #test_set
        for root, _, files in os.walk(os.path.join(path_vox1, 'test')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)

                    item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_items.append(item)

        # test_set_E
        self.test_items_E = self.test_items.copy()
        for root, _, files in os.walk(os.path.join(path_vox1, 'train')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_items_E.append(item)

        # test_items_noise
        for noise in noises:
            for root, _, files in os.walk(os.path.join(path_vox1, f'test_noise/{noise}')):
                for file in files:
                    if '.wav' in file:
                        f = os.path.join(root, file)
                        item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                        self.test_items_noise[noise].append(item)

        
        self.trials = self.parse_trials(os.path.join(path_vox1, 'trials/trials.txt'))
        self.trials_E = self.parse_trials(os.path.join(path_vox1, 'trials/trials_E.txt'))
        self.trials_H = self.parse_trials(os.path.join(path_vox1, 'trials/trials_H.txt'))

        # error check
        assert len(self.train_items) == NUM_TRAIN_ITEM
        assert len(self.test_items_E) == NUM_TEST_ITEM
        assert len(self.trials) == NUM_TRIALS
        assert len(self.trials_E) == NUM_TRIALS_E
        assert len(self.trials_H) == NUM_TRIALS_H
        assert len(labels) == NUM_TRAIN_SPK


    def parse_trials(self, path):
        trials = []
        for line in open(path).readlines():
            strI = line.split(' ')
            item = Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            trials.append(item)
        return trials