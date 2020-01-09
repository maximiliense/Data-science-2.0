from sklearn.model_selection import train_test_split

from datascience.data.datasets import ImageDatasetMTBernoulli
from datascience.data.util.source_management import check_source
from datascience.model_selection.util import perform_split
from engine.logging import print_dataset_statistics

from torchvision import transforms

import os


def load_multitask_bernoulli_dataset(source, test_size=0.1, val_size=0.1, transform=None, splitter=train_test_split):
    r = check_source(source)
    path = r['path']
    classes_index = {}
    labels_index = {}
    labels = []
    classes = []
    images = []

    for c in os.listdir(path):
        classes_index[c] = len(classes_index)  # to index the classes
        path_class = os.path.join(path, c)
        for label in os.listdir(path_class):
            path_label = os.path.join(path_class, label)

            # it is a bernoulli task and there must be only two labels: positive and negative
            # the name should be shared among classes
            if label not in labels_index:
                if len(labels_index) >= 2:
                    raise PosNegLabelException('All positive and negative labels folder must have the same name...')
                labels_index[label] = len(labels_index)
            for image in os.listdir(path_label):
                labels.append((label, labels_index[label]))  # label name, label ID
                classes.append((c, classes_index[c]))  # class name, class ID
                images.append(os.path.join(path_label, image)) # image path
    dataset = (labels, classes, images)

    # dataset split
    train, test = perform_split(dataset, test_size, splitter)
    train, val = perform_split(train, val_size, splitter)

    if transform is None:
        transform = {'train': transforms.Compose([transforms.ToTensor()]),
                     'test': transforms.Compose([transforms.ToTensor()])}
    train = ImageDatasetMTBernoulli(train, len(classes_index), transform=transform['train'])
    val = ImageDatasetMTBernoulli(val, len(classes_index), transform=transform['test'])
    test = ImageDatasetMTBernoulli(test, len(classes_index), transform=transform['test'])

    print_dataset_statistics(
        len(train), len(val), len(test), source, len(classes_index)
    )
    return train, val, test


class PosNegLabelException(Exception):
    def __init__(self, message):
        super(PosNegLabelException, self).__init__(message)
