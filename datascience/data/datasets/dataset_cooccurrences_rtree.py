import torch
from torch.utils.data import Dataset
import numpy as np
from engine.dataset.rasters.environmental_raster_glc import PatchExtractor


class GeoLifeClefDataset(Dataset):
    def __init__(self, root_dir, labels, dataset, ids, n_neighbours=200, extractor=None, nb_labels=3336):
        self.labels = labels
        self.ids = ids
        self.dataset = dataset
        self.n_neighbours = n_neighbours
        self.nb_labels = nb_labels

        self.kdtree = None
        self.train_dataset = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pos = [list(self.dataset[idx])]
        dist, neighbours = self.kdtree.query(pos, k=self.n_neighbours)
        coocs = np.zeros(self.nb_labels)
        for n in neighbours[0]:
            if idx != self.train_dataset.ids[n]:
                coocs[self.train_dataset.labels[n]] += 1

        return torch.from_numpy(coocs).float(), self.labels[idx]
