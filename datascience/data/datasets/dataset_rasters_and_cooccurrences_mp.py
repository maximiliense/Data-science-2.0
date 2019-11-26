import torch
from torch.utils.data import Dataset
from datascience.data.rasters.environmental_raster_glc import PatchExtractor


class GeoLifeClefDataset(Dataset):
    def __init__(self, root_dir, labels, dataset, ids, extractor=None, nb_labels=3336, second_neihbour=True):
        self.labels = labels
        self.ids = ids
        self.dataset = dataset
        self.nb_labels = nb_labels
        self.extractor = extractor

        self.pos_multipoints = None
        self.kdtree = None
        self.multipoints = None
        self.second_neihbour = second_neihbour

        if extractor is None:
            self.extractor = PatchExtractor(root_dir, size=64, verbose=True)
            self.extractor.add_all()
        else:
            self.extractor = extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pos = [list(self.dataset[idx])]
        dist, neighbours = self.kdtree.query(pos, k=2)
        if dist[0][0] == 0 and self.second_neihbour:
            mp_pos = self.pos_multipoints[neighbours[0][1]]
        else:
            mp_pos = self.pos_multipoints[neighbours[0][0]]
        coocs = self.multipoints[mp_pos]

        tensor = self.extractor[self.dataset[idx]]

        return (torch.from_numpy(tensor).float(), torch.from_numpy(coocs).float()), self.labels[idx]
