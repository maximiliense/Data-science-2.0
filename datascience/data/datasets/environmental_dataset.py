import torch
from torch.utils.data import Dataset

import numpy as np

from datascience.data.rasters.environmental_raster_glc import PatchExtractor
from engine.flags import deprecated


class EnvironmentalDataset(Dataset):
    def __init__(self, labels, dataset, ids, rasters, size_patch=64, extractor=None, transform=None, add_all=True):
        self.labels = labels
        self.ids = ids
        self.dataset = dataset
        if extractor is None:
            self.extractor = PatchExtractor(rasters, size=size_patch, verbose=True)
            if add_all:
                self.extractor.add_all()
        else:
            self.extractor = extractor

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if type(self.extractor) is not bool:
            tensor = self.extractor[self.dataset[idx]]
            if self.transform is not None:
                tensor = self.transform(tensor).copy()
            return torch.from_numpy(tensor).float(), self.labels[idx]
        else:
            return self.dataset[idx], self.labels[idx]

    def numpy(self):
        return np.array([self[i][0] for i in range(len(self))]), self.labels

    @deprecated()
    def get_vectors(self):
        vec = []
        for idx, data in enumerate(self.dataset):
            vector = self.extractor[self.dataset[idx]]
            if self.transform is not None:
                vector = self.transform(vector).copy()
            vector = list(vector)
            vec.append(vector)
        return vec

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__ + '(size: {})'.format(len(self))
