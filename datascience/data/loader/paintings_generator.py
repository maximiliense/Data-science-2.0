import os
from sklearn.model_selection import train_test_split
import random

from torchvision import transforms

from datascience.data.datasets.image_dataset import ImageDataset
from datascience.data.util.source_management import check_source


class PaintingDatasetGenerator(object):
    """
    generates multiple types of datasets
    """
    def __init__(self, source, transform=None, input_size=299):
        r = check_source(source)
        path = r['path']
        if transform is None:
            self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            self.test_transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        dataset = _load_dataset(path)
        random.shuffle(dataset)

        self.country = []
        self.painter = []
        self.type = []
        self.path = []

        for row in dataset:
            self.country.append(row[0])
            self.painter.append(row[1])
            self.type.append(row[2])
            self.path.append(
                os.path.join(*row))

    def unique_painters(self):
        """
        :return: an array of strings (i.e. names of the painters)
        """
        painters_list = []
        index = {}
        for p in self.painter:
            if p not in index:
                index[p] = True
                painters_list.append(p)
        return painters_list

    def painter_dataset(self, test_size=0.1, val_size=0.1):
        """
        creates a dataset where the label is the painter himself
        :param test_size:
        :param val_size:
        :return: the dataset
        """
        index = {}
        for p in self.painter:
            if p not in index:
                index[p] = len(index)

        dataset = [(self.path[i], index[self.painter[i]]) for i in range(len(self.path))]

        train, test = train_test_split(dataset, test_size=test_size)
        train, val = train_test_split(
            train, test_size=val_size) if val_size > 0. else (train, None)

        return (ImageDataset(train, self.train_transform),
                ImageDataset(val, self.test_transform),
                ImageDataset(test, self.test_transform))

    def country_dataset_one_fold(self, painter_val, painter_test):
        """
        creates a data set where all painting are added in the training set, except those from painter_val
        and painter_test
        :param painter_val:
        :param painter_test:
        :return: the dataset
        """
        dataset = {
            'train': [],
            'val': [] if painter_val is not None else None,
            'test': []
        }
        index = {}
        for i in range(len(self.path)):
            if self.country[i] not in index:
                index[self.country[i]] = len(index)
            if self.painter[i] == painter_val:
                dataset['val'].append((self.path[i], index[self.country[i]]))
            elif self.painter[i] == painter_test:
                dataset['test'].append((self.path[i], index[self.country[i]]))
            else:
                dataset['train'].append((self.path[i], index[self.country[i]]))
        return (ImageDataset(dataset['train'], self.train_transform),
                ImageDataset(dataset['val'], self.test_transform),
                ImageDataset(dataset['test'], self.test_transform))

    def country_dataset(self, test_size=0.1, val_size=0.1):
        """
        construct a dataset where the label is the country
        :param test_size:
        :param val_size:
        :return:
        """
        index_split, dataset, index_labels = self._configure_dataset_creation(test_size, val_size)
        return self._construct_dataset(index_split, dataset, index_labels, self.type)

    def type_dataset(self, test_size=0.1, val_size=0.1):
        """
        construct a dataset where the label is the painting type
        :param test_size:
        :param val_size:
        :return:
        """
        index_split, dataset, index_labels = self._configure_dataset_creation(test_size, val_size)
        return self._construct_dataset(index_split, dataset, index_labels, self.type)

    def _configure_dataset_creation(self, test_size=0.1, val_size=0.1):
        index_split = self._painter_split(None, test_size, val_size)
        dataset = {
            'train': [],
            'val': [] if val_size > 0. else None,
            'test': []
        }
        index_labels = {}
        return index_split, dataset, index_labels

    def _construct_dataset(self, index_split, dataset, index_labels, labels):
        for i in range(len(self.path)):
            if labels not in index_labels:
                index_labels[labels] = len(index_labels)
            dataset[index_split[self.painter[i]]].append((
                self.path[i], index_labels[labels]
            ))
        return (ImageDataset(dataset['train'], self.train_transform),
                ImageDataset(dataset['val'], self.test_transform),
                ImageDataset(dataset['test'], self.test_transform))

    def _painter_split(self, painters_list=None, test_size=0.1, val_size=0.1):
        if painters_list is None:
            index = {}
            for p in self.painter:
                if p not in index:
                    index[p] = len(index)
            painters_list = list(index.keys())
        train, test = train_test_split(painters_list, test_size=test_size)
        train, val = train_test_split(train, test_size=val_size) if val_size > 0. else (train, None)

        index = {}
        for p in train:
            index[p] = 'train'
        if val is not None:
            for p in val:
                index[p] = 'val'
        for p in test:
            index[p] = 'test'

        return index


def _load_dataset(path):
    """
    The structure must be country/painter/painting_type/painting.jpg.
    The type must be either jpg or jpeg (all cases accepted).
    :param path:
    :return:
    """
    dataset = []

    for country in os.listdir(path):
        _path = os.path.join(path, country)
        if not os.path.isfile(_path):
            for painter in os.listdir(_path):
                _path = os.path.join(path, country, painter)
                if not os.path.isfile(_path):
                    for painting_type in os.listdir(_path):
                        _path = os.path.join(path, country, painter, painting_type)
                        if not os.path.isfile(_path):
                            for painting in os.listdir(_path):
                                _path = os.path.join(
                                    path, country, painter, painting_type, painting
                                )
                                _, ext = os.path.splitext(_path)
                                if ext.lower() in ('.jpg', '.jpeg'):
                                    dataset.append((
                                        country, painter, painting_type, _path)
                                    )
    return dataset
