from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        :param dataset:
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = Image.open(self.dataset[idx][0]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, self.dataset[idx][1]


class ImageDatasetMTBernoulli(Dataset):
    def __init__(self, dataset, nb_classes, transform=None):
        """
        :param dataset:
        """
        self.labels = dataset[0]
        self.classes = dataset[1]
        self.images = dataset[2]
        self.transform = transform
        self.nb_classes = nb_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        labels = torch.from_numpy(np.array(
            [1 if self.classes[idx][1] == i else 0 for i in range(self.nb_classes)]
        )).float()

        return img, labels
