import os

from datascience.ml.neural.models.util import one_label
from engine.core import module
from engine.hardware import use_gpu, first_device, all_devices, device_description
from engine.logging import print_info, print_errors

import torch

from engine.parameters.special_parameters import from_scratch
from engine.path import output_path


_checkpoint_path = 'models/{}.torch'

_model = None
_optimizer = None

_checkpoint = {}


@module
def create_model(model_class, model_params, model_name='model', p_label=one_label):
    """
    create and eventually load model
    :param model_name:
    :param model_class:
    :param model_params:
    :param model_name:
    :param p_label:
    :return:
    """
    model = model_class(**model_params)

    if not from_scratch:  # recover from checkpoint
        _load_model(model, model_name)

    # configure usage on GPU
    if use_gpu():
        model.to(first_device())
        model = torch.nn.DataParallel(model, device_ids=all_devices())

    model.p_label = p_label

    # print info about devices
    print_info('Device(s)): ' + str(device_description()))

    return model


def create_optimizer(parameters, optimizer_class, optim_params, model_name='model'):
    """
    create and eventually load optimizer
    :param model_name:
    :param parameters:
    :param optimizer_class:
    :param optim_params:
    :return:
    """
    opt = optimizer_class(parameters, **optim_params)
    if not from_scratch:
        _load_optimizer(opt, model_name)
    return opt


def _load_optimizer(optimizer, model_name):
    """
    load checkpoint
    :param optimizer:
    :return:
    """
    global _checkpoint
    if model_name not in _checkpoint:
        _load_checkpoint(model_name)

    if 'optimizer_state_dict' in _checkpoint[model_name]:
        optimizer.load_state_dict(_checkpoint[model_name]['optimizer_state_dict'])


def _load_model(model, model_name):
    """
    load checkpoint
    :param model:
    :param model_name:
    :return:
    """
    global _checkpoint
    if model_name not in _checkpoint:
        _load_checkpoint(model_name)

    if 'model_state_dict' in _checkpoint[model_name]:
        model.load_state_dict(_checkpoint[model_name]['model_state_dict'])
    else:
        model.load_state_dict(_checkpoint[model_name])


def _load_checkpoint(model_name):
    path = output_path(_checkpoint_path.format(model_name), have_validation=True)

    global _checkpoint
    if not os.path.isfile(path):
        print_errors('{} does not exist'.format(path), do_exit=True)
    print_info('Loading checkpoint from ' + path)
    _checkpoint[model_name] = torch.load(path)


def save_checkpoint(model, optimizer=None, model_name='model', validation_id=None):
    """
    save checkpoint (optimizer and model)
    :param model_name:
    :param validation_id:
    :param model:
    :param optimizer:
    :return:
    """
    path = output_path(_checkpoint_path.format(model_name), validation_id=validation_id, have_validation=True)

    print_info('Saving checkpoint: ' + path)

    model = model.module if type(model) is torch.nn.DataParallel else model

    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)
