import torch
from torch.utils.data import Dataset
from datascience.data.rasters.environmental_raster_glc import PatchExtractor
from datascience.data.transforms import random_rotation
from engine.util.console.flags import deprecated


@deprecated(comment='Transforms should be attribute of the regular EnvironmentalDataset.')
class GeoLifeClefDataset(Dataset):
    def __init__(self, root_dir, labels, dataset, ids, extractor=None):
        self.extractor = extractor
        self.labels = labels
        self.ids = ids
        self.dataset = dataset
        if extractor is None:
            self.extractor = PatchExtractor(root_dir, size=64, verbose=True)
            self.extractor.add_all()
        else:
            self.extractor = extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tensor = self.extractor[self.dataset[idx]]
        tensor = random_rotation(tensor).copy()
        return torch.from_numpy(tensor).float(), self.labels[idx]
