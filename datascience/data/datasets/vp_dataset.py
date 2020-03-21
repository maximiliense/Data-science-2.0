from torch.utils.data import Dataset
import torch
import numpy as np


def generate_image(patterns, patterns_ratio):
    pattern_size = 2
    image_size = 64
    if sum(patterns_ratio) > 1:
        raise Exception('pattern_1_ratio + pattern_2_ratio \\in [0,1]')
    rows = []
    for i in range(int(image_size/pattern_size)):
        row = []
        for j in range(int(image_size/pattern_size)):
            i = 0
            r = np.random.random()
            while i < len(patterns) and sum(patterns_ratio[:i+1]) < r:
                i += 1
            if i < len(patterns):
                row.append(patterns[i])
            else:
                row.append(np.random.random((2, 2)))
        rows.append(np.concatenate(row))
    image = np.concatenate(rows, axis=1)

    return image-0.5


class VisualPatternDataset(Dataset):
    def __init__(self, dataset_size, patterns, patterns_ratio, label_ratio):
        self.labels = [0 if np.random.random() < label_ratio else 1 for _ in range(dataset_size)]
        self.dataset = [np.expand_dims(generate_image(patterns, patterns_ratio[lab]), axis=0) for lab in self.labels]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.from_numpy(self.dataset[idx]).float(), self.labels[idx]
