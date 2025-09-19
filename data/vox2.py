import os
from dataclasses import dataclass

NUM_TRAIN_ITEM = 1092009
NUM_TRAIN_SPK = 5994

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

class VoxCeleb2:
    def __init__(self, path_vox2):
        self.train_items = []
        self.class_weight = []

        # train_set
        labels = {}
        num_utt = [0 for _ in range(NUM_TRAIN_SPK)]
        num_sample = 0
        for root, _, files in os.walk(os.path.join(path_vox2, 'train')):
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