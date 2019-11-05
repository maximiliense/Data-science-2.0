import torch

from torch.autograd import Variable
from engine.hardware import use_gpu, first_device, all_devices, device_description
from engine.path import output_path
from engine.logging import print_info, print_errors

import os


def one_input(x):
    return x if x.requires_grad else Variable(x),


def first_label(l):
    return Variable(l[0].cuda()) if use_gpu() else Variable(l[0])


def multi_labels(l):
    r = tuple(Variable(label.cuda()) for label in l) if use_gpu() else tuple(Variable(label) for label in l)
    return r


def multi_inputs(l):
    return tuple(Variable(label) for label in l)


def one_label(l):
    return Variable(l.cuda()) if use_gpu() else Variable(l)


def decorated_forward(fn, p_input):
    def new_forward(self, _input):
        return fn(self, *p_input(_input))
    return new_forward


def set_model_last_layer(model, last_layer):
    # set model last_layer arg and return old value
    old_last_layer = None
    if hasattr(model, 'last_layer'):
        old_last_layer = model.last_layer
        model.last_layer = last_layer
    elif type(model) is torch.nn.DataParallel and hasattr(model.module, 'last_layer'):
        old_last_layer = model.module.last_layer
        model.module.last_layer = last_layer
    return old_last_layer


def set_model_logit(model, logit):
    # set model ml arg and return old value
    old_logit = None
    if hasattr(model, 'logit'):
        old_logit = model.logit
        model.logit = logit
    elif type(model) is torch.nn.DataParallel and hasattr(model.module, 'logit'):
        old_logit = model.module.logit
        model.module.logit = logit
    return old_logit
