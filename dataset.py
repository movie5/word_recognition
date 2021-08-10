import os
import torch
from glob import glob
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data.dataset import Dataset
import torchaudio
from audio_augmentations import *
from time import time
from utils import *



class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=False)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class small_dataset(Dataset):
    def __init__(self, root_dir, mode, hard_split):
        self.root_dir = root_dir
        self.class_list = os.listdir(root_dir)
        self.x = []
        self.y = []
        self.mode = mode
        self.hard_split = hard_split
        count = 0
        for item in self.class_list:
            item_dir = os.path.join(self.root_dir, item)
            x_list = glob(item_dir + '/*.wav')
            y_list = [item]*len(x_list)
            for x, y in zip(x_list, y_list):
                if count == 0:
                    audio = torchaudio.load(x)[0]
                    self.x = (audio - torch.mean(audio)) / torch.max(torch.abs(audio))
                else:
                    try:
                        audio = torchaudio.load(x)[0]
                        self.x = torch.vstack((self.x, (audio - torch.mean(audio)) / torch.max(torch.abs(audio))))
                    except:
                        continue
                if self.hard_split:
                    self.y = self.y + [(y, x.split('_')[-1].split('.')[0])]
                elif self.mode == 'human':
                    self.y = self.y + [x.split('_')[-1].split('.')[0]]
                elif self.mode == 'word':
                    self.y = self.y + [y]
                count += 1
        if not(self.hard_split):
            self.labels = sorted(list(set(data for data in self.y)))
        else:
            if self.mode == 'word':
                self.labels = sorted(list(set(data[0] for data in self.y)))
            elif self.mode == 'human':
                self.labels = sorted(list(set(data[1] for data in self.y)))

        self.sr = torchaudio.load(x)[1]
        self.new_sr = 1000
        self.resample = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self.new_sr)
        self.x = self.resample(self.x)
        self.transforms_aug = [
            RandomApply([Noise(min_snr=0.1, max_snr=0.2)], p=0.5),
            RandomApply([Gain()], p=0.5),
        ]
        self.transform_aug = Compose(transforms= self.transforms_aug)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.transform_aug(torch.reshape(self.x[idx], [1, self.x[idx].shape[0]])), self.y[idx], self.sr 

    def remove(self):
        temp = []
        for item in self.y:
            if self.mode == 'word':
                temp.append(item[0])
            elif self.mode == 'human':
                temp.append(item[1])
        self.y = temp

    def to_index(self):
        self.y = [label_to_index(self.labels, item) for item in self.y]
        return self.labels