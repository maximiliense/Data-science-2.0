from torch.utils.data import Dataset


class GeoLifeClefDataset(Dataset):
    def __init__(self, labels, dataset, ids, rasters,  extractor=None, **kwargs):
        """
        :param dataset:
        :param labels:
        :param ids:
        """
        self.labels = labels
        self.dataset = dataset
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def get_vectors(self):
        return self.dataset
