import os

import numpy as np
import torch
from torch.utils.data import Dataset

class BiologicalDataset(Dataset):
    def __init__(self, train=True):
        # load all dataset in memory since it's lightweight
        if train:
            self.X = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'X_train.npy'))
            self.y = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'real_classes_train_int.npy'))
        else:
            self.X = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'X_test.npy'))
            self.y = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'real_classes_test_int.npy'))

        self.X = np.transpose(self.X, (0, 3, 1, 2))       # channel first
        self.X = torch.as_tensor(self.X, dtype=torch.float32)
        self.y = torch.as_tensor(self.y, dtype=torch.int64)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)