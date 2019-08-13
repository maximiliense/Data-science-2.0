from torch.utils.data import Dataset


class GeoLifeClefDataset(Dataset):
    def __init__(self, _, labels, dataset, ids, extractor=None):
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
        return self.dataset[idx], self.dataset[idx]
