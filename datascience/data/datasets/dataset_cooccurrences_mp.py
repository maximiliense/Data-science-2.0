import torch
from torch.utils.data import Dataset
import numpy as np
from engine.dataset.rasters.environmental_raster_glc import PatchExtractor


class GeoLifeClefDataset(Dataset):
    def __init__(self, root_dir, labels, dataset, ids, extractor=None, nb_labels=3336):
        self.labels = labels
        self.ids = ids
        self.dataset = dataset
        self.nb_labels = nb_labels

        self.pos_multipoints = None
        self.kdtree = None
        self.multipoints = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pos = [list(self.dataset[idx])]
        dist, neighbours = self.kdtree.query(pos, k=2)
        if dist[0][0] == 0:
            mp_pos = self.pos_multipoints[neighbours[0][1]]
        else:
            mp_pos = self.pos_multipoints[neighbours[0][0]]
        coocs = self.multipoints[mp_pos]

        return torch.from_numpy(coocs).float(), self.labels[idx]
