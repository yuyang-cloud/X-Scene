import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class F2DDataset(Dataset):
    def __init__(self, trainset_path=None, valset_path=None, max_files=None, return_dict=False):
        self.return_dict = return_dict
        self.files = []
        if trainset_path is not None:
            self.files_trainset = [os.path.join(trainset_path, f) for f in os.listdir(trainset_path) if f.endswith('.npy')]
            self.files.extend(self.files_trainset)
        if valset_path is not None:
            self.files_valset = [os.path.join(valset_path, f) for f in os.listdir(valset_path) if f.endswith('.npy')]
            self.files.extend(self.files_valset)
        random.shuffle(self.files)
        if max_files is not None:
            self.files = self.files[:max_files]
        if len(self.files) == 0:
            raise FileNotFoundError("No .npy files were found for F2DDataset.")
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]
        )
        self.files.sort(key=lambda x: self._get_token(x))
        self.tokens = [self._get_token(f) for f in self.files]

    @staticmethod
    def _get_token(path):
        return Path(path).stem

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx])
        img = torch.tensor(img).permute(2, 0, 1).to(torch.uint8)
        img = self.transform(img)
        if self.return_dict:
            return {
                'img': img,
                'path': self.files[idx],
            }
        else:
            return img
