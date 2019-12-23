from torch.utils.data import Dataset
from PIL import Image


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
