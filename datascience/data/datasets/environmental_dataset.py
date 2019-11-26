import torch
from torch.utils.data import Dataset

import numpy as np

from datascience.data.rasters.environmental_raster_glc import PatchExtractor
from engine.flags import deprecated


ENVIRONMENTAL_DATASET_EXTRACTOR = None


class EnvironmentalDataset(Dataset):
    def __init__(self, labels, dataset, ids, rasters, size_patch=64, transform=None,
                 add_all=True, limit=-1, reset_extractor=False):
        self.labels = labels
        self.ids = ids
        self.dataset = dataset

        self.limit = limit
        global ENVIRONMENTAL_DATASET_EXTRACTOR
        if ENVIRONMENTAL_DATASET_EXTRACTOR is None or reset_extractor:
            self.extractor = PatchExtractor(rasters, size=size_patch, verbose=True)
            if add_all:
                self.extractor.add_all()
            ENVIRONMENTAL_DATASET_EXTRACTOR = self.extractor
        else:
            self.extractor = ENVIRONMENTAL_DATASET_EXTRACTOR

        self.transform = transform

    def __len__(self):
        return len(self.labels) if self.limit == -1 else min(len(self.labels), self.limit)

    def __getitem__(self, idx):
        if type(self.extractor) is not bool:
            tensor = self.extractor[self.dataset[idx]]
            if self.transform is not None:
                tensor = self.transform(tensor).copy()
            return torch.from_numpy(tensor).float(), self.labels[idx]
        else:
            return self.dataset[idx], self.labels[idx]

    def numpy(self):
        """
        :return: a numpy dataset of 1D vectors
        """
        return np.array([torch.flatten(self[i][0]).numpy() for i in range(len(self))]), self.labels

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
