import torch.nn as nn
from torch.nn import DataParallel

from engine.core import module
from engine.parameters import special_parameters
import torch

import numpy as np


@module
def extract_representation(dataset, model, batch_size=32, labels_index=None):
    """
    returns the representation learnt by the model.
    :param labels_index:
    :param dataset:
    :param model: The last layer of the model must be the attribute called fc
    :param batch_size:
    :return:
    """

    fc = model.module.fc if type(model) is DataParallel else model.fc
    model.fc = nn.Sequential()
    extraction = []
    labels_ext = []
    labels_str = []

    batch_size = len(dataset) if batch_size == -1 else batch_size
    num_workers = special_parameters.nb_workers
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size,
                                              num_workers=num_workers)

    model.eval()

    for idx, data in enumerate(data_loader):
        inputs, labels = data
        outputs = model(inputs)
        extraction.append(outputs.detach().cpu().numpy())
        labels_ext.append(labels.detach().cpu().numpy())

    if type(model) is DataParallel:
        model.module.fc = fc
    else:
        model.fc = fc
    labels_ext = np.concatenate(labels_ext)
    if labels_index is not None:
        for i in labels_ext:
            labels_str.append(labels_index[i])
        return np.concatenate(extraction), labels_ext, labels_str
    else:
        return np.concatenate(extraction), labels_ext
