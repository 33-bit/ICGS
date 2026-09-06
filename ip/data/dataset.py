"""Original file-backed dataset and augmentation behavior."""

import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import torch
from torch.utils.data import Dataset


class RunningDataset(Dataset):
    def __init__(self, data_path, num_samples, rec=False, rand_g_prob=0.0, random_rotation=False):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.random_rotation = random_rotation
        self.rec = rec
        if rec:
            self.data_attr = [
                'pos',
                'queries',
                'batch_queries',
                'batch_pos',
                'occupancy',
            ]
        else:
            self.data_attr = [
                'actions',
                'actions_grip',
            ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:
            try:
                data = torch.load(os.path.join(self.data_path, 'data_{}.pt'.format(idx)))
                for attr in self.data_attr:
                    assert hasattr(data, attr)

                if np.random.uniform() < self.rand_g_prob:
                    data.current_grip *= -1

                if self.random_rotation and self.rec:
                    R = torch.tensor(Rot.random().as_matrix(), dtype=data.pos.dtype, device=data.pos.device)
                    data.pos = data.pos @ R.T
                    data.queries = data.queries @ R.T
                return data
            except Exception as e:
                idx = np.random.randint(0, self.num_samples)
