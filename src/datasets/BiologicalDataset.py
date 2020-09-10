import numpy as np
import os

from torch.utils.data import Dataset

class BiologicalDataset(Dataset):
    def __init__(self, train=True):
        if train:
            self.X = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'X_train.npy'))
            self.y = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'real_classes_train_int.npy'))
        else:
            self.X = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'X_test.npy'))
            self.y = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'real_classes_test_int.npy'))

        self.X = np.transpose(self.X, (0, 3, 1, 2))       # channel first
        self.X = self.X.astype('float32')

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)