import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
from numpy.random import normal


def sample_points(label, gaussian=False, sample_params=None):
    if sample_params is None or 'scale' not in sample_params:
        sample_params = {'scale': 0.5}
    if not gaussian:
        return np.random.uniform(0, np.pi if label == 0 else -np.pi)

    else:
        return normal(np.pi/2. if label == 0 else -np.pi/2., **sample_params)


def create_dataset(n_red=20, n_blue=20, gaussian=False, sample_params=None):
    dataset = []
    for _ in range(n_blue):
        angle = sample_points(0, gaussian, sample_params)
        dataset.append((np.cos(angle), np.sin(angle), 0))

    for _ in range(n_red):
        angle = sample_points(1, gaussian, sample_params)
        dataset.append((np.cos(angle), np.sin(angle), 1))
    return np.array(dataset)


class Dataset(TorchDataset):
    def __init__(self, n_blue=20, n_red=20, gaussian=False, sample_params=None):
        dataset = create_dataset(n_blue, n_red, gaussian=gaussian, sample_params=sample_params)
        self.dataset = dataset[:, :2]
        self.labels = dataset[:, 2].astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.dataset[idx]).float(), self.labels[idx]
