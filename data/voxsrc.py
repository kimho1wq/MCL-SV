import os
from dataclasses import dataclass

NUM_TRIALS_SRC_23 = 49987
NUM_TRIALS_SRC_22 = 306432

@dataclass
class EnrollmentItem:
    key: str
    path: str

@dataclass
class Trial:
    key1: str
    key2: str
    label: int

class VoxSRC23:
    def __init__(self, path_voxsrc23):
        self.items = []
        self.trials = []
                    
        # test_set_src23
        for root, _, files in os.walk(path_voxsrc23):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = EnrollmentItem(path=f, key=f.split('/')[-1])
                    self.items.append(item)

        
        self.trials = self.parse_trials(os.path.join(path_voxsrc23, 'VoxSRC2023_val.txt'))
        

        # error check
        assert len(self.trials) == NUM_TRIALS_SRC_23
        
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
