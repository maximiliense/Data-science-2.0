import os

import torch
from torch.utils.data import Dataset
import numpy as np

from datascience.data.rasters.environmental_raster_glc import PatchExtractor


class EnvironmentalIGNDataset(Dataset):
    def __init__(self, labels, dataset, ids, rasters, patches, size_patch=64, extractor=None, transform=None,
                 add_all=True, limit=-1):
        self.extractor = extractor
        self.labels = labels
        self.ids = ids
        self.dataset = dataset
        self.patches = patches

        self.limit = limit

        if extractor is None:
            self.extractor = PatchExtractor(rasters, size=size_patch, verbose=True)
            if add_all:
                self.extractor.add_all()
        else:
            self.extractor = extractor

        self.transform = transform

    def file_exists(self, idx):
        return os.path.isfile(self.path(idx))

    def path(self, idx):
        image_id = str(int(self.ids[idx]))
        return os.path.join(self.patches, image_id[-2:], image_id[-4:-2], image_id + '.npy')

    def __len__(self):
        return len(self.labels) if self.limit == -1 else min(len(self.labels), self.limit)

    def __getitem__(self, idx):

        ign_patch = np.transpose(np.load(self.path(idx)), (2, 0, 1))

        tensor = self.extractor[self.dataset[idx]]
        tensor = np.concatenate([tensor, ign_patch], axis=0)
        if self.transform is not None:
            tensor = self.transform(tensor).copy()
        return torch.from_numpy(tensor).float(), self.labels[idx]

    @property
    def named_dimensions(self):
        return [r.name for r in self.extractor.rasters] + ['IGN']

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__ + '(size: {})'.format(len(self))
