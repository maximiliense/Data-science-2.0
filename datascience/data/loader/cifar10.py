import os

import torchvision
import torchvision.transforms as transforms

from engine.parameters.special_parameters import homex
from engine.core import module
from engine.util.console.logs import print_dataset_statistics


@module
def cifar():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_root = os.path.join(homex, 'data')

    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                             download=True, transform=transform)

    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                            download=True, transform=transform)

    print_dataset_statistics(len(train_set), 0, len(test_set), 'CIFAR10', 10)

    return train_set, test_set
