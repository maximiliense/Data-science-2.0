import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

np.random.seed(42)


def create_dataset(N=100, error=True):
    x = np.array(np.linspace(-0.5, 0.5, 100))
    y = (np.power(x, 2) * 5 + 18 * np.power(2*x, 3) + x + 2) / 15
    probability_error = 0.3
    error_threshold = 0.5
    dataset = []
    label = []

    while len(label) < N:
        obs_x = np.random.rand() - 0.5
        obs_y = (np.random.rand()*35 - 15) / 15
        f_value = (5 * obs_x ** 2 + 18 * (2*obs_x) ** 3 + obs_x + 2) / 15
        if error and np.random.rand() <= probability_error:
            sign = +1 if np.random.rand() > 0.5 else -1
            label.append(int(f_value + sign * error_threshold > obs_y))
        else:
            label.append(int(f_value > obs_y))
        dataset.append([obs_x, obs_y])

    return np.array(dataset), np.array(label), np.array([x, y])


class Dataset(TorchDataset):
    def __init__(self, size=100, error=True):
        self.dataset, self.label, self.separator = create_dataset(size, error=error)
        self.labels = self.label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return torch.from_numpy(self.dataset[idx]).float(), self.label[idx]
