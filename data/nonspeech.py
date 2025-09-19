import os
from dataclasses import dataclass


NUM_TRAIN_SPK = 1211
NUM_TRIALS = 37611
NUM_TEST_ITEM = 4874


@dataclass
class EnrollmentItem:
    key: str
    path: str

@dataclass
class Trial:
    key1: str
    key2: str
    label: int

class NonSpeech:
    def __init__(self,  path_nonspeech):
        SNR = [0,5,10,15,20]
        self.items = { i : [] for i in SNR}
        self.trials = []

        # test_set
        for snr in SNR:
            for root, _, files in os.walk(f'{path_nonspeech}_{snr}'):
                for file in files:
                    if '.wav' in file:
                        f = os.path.join(root, file)
                        item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                        self.items[snr].append(item)

       
        