import os
from dataclasses import dataclass

NUM_TRIALS = 159536
NUM_TEST_ITEMS = 7660+4874

@dataclass
class EnrollmentItem:
    key: str
    path: str

@dataclass
class Trial:
    key1: str
    key2: str
    label: int

class VCMix:
    def __init__(self, path_vcmix, path_vox1):
        self.items = []
        self.trials = []
        
        # test_set
        for root, _, files in os.walk(os.path.join(path_vox1, 'test')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.items.append(item)
                    
        for root, _, files in os.walk(path_vcmix):
            for file in files:
                if '.wav' in file: 
                    f = os.path.join(root, file)
                    item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-2:]))
                    self.items.append(item)

        
        self.trials = self.parse_trials(os.path.join(path_vcmix, 'vcmix_test.txt'))
        

        # error check
        assert len(self.trials) == NUM_TRIALS
        assert len(self.items) == NUM_TEST_ITEMS
        
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