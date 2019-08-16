from random import choice, randint

import numpy as np

import torch
from torch.nn import DataParallel


def _optimizable(model, is_optimizable=True):
    for p in model.parameters():
        p.require_grad = is_optimizable


def compute_filters(model, data, nb_elements=10, nb_filters=10, include_logit=False):
    """
    compute the filters at multiple levels in multiple partitions. Notice that for convolutional filters,
    the code guarantees that all filters from the same layer will concern the same area of the input tensor. Thus
    scalar product and other distance can be applied between gradient
    :param model: the model on which to compute the filters
    :param data: some data that will be used to compute the filters
    :param nb_elements: the number of elements from data to use to compute the filters
    :param nb_filters: the number of filters to compute for each element in data
    :param include_logit: include the logit when computing the filters
    :return: the various filters in a list of shape [list of layers, list of filter in the layer]
    """
    include_logit = 1 if include_logit else 0

    model_instance = model.module if type(model) is DataParallel else model

    # the element that will be used
    batch = torch.stack([choice(data)[0] for _ in range(nb_elements)])

    # should be optimizable
    batch.requires_grad = True

    # the model parameters should not be optimizable
    _optimizable(model, False)

    # dimension of an image.. the gradient should have the same dimension
    dimension_grad = batch[0].shape[0] * batch[0].shape[1] * batch[0].shape[2]

    filters = []

    for layer in range(1, len(model_instance) + include_logit):
        output = model(batch, layer=layer)

        layer_filters = []
        filters.append(layer_filters)

        for j in range(nb_filters):
            # if convolutional filter, a single filter produces multiple outputs..
            # we have to select one
            select_filter_output = 0
            if j == 0 and len(output.shape) > 2:
                select_filter_output = randint(0, output.shape[2] - 1)

            # select one of the filters randomly
            select_filter = randint(0, output.shape[1] - 1)
            for i in range(nb_elements):
                found_filter = False
                c = 0
                if batch.grad is not None:
                    batch.grad.data.zero_()

                if len(output.shape) > 2:
                    output[i, select_filter, select_filter_output].backward(retain_graph=True)
                else:
                    output[i, select_filter].backward(retain_graph=True)

                layer_filters.append(torch.flatten(batch.grad[i].detach().cpu()).numpy())

    _optimizable(model)
    for i in range(len(filters)):
        filters[i] = np.array(filters[i])
    return filters
