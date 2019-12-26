import torch.nn as nn

from engine.core import module
from engine.parameters import special_parameters
import torch

import numpy as np


@module
def extract_representation(dataset, model, batch_size=32):
    """
    returns the representation learnt by the model.
    :param dataset:
    :param model: The last layer of the model must be the attribute called fc
    :param batch_size:
    :return:
    """
    fc = model.fc
    model.fc = nn.Sequential()
    extraction = []
    labels_ext = []

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

        if idx == 3:
            break

    model.fc = fc
    return np.concatenate(extraction), np.concatenate(labels_ext)
