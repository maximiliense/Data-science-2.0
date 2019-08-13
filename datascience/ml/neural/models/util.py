import torch

from torch.autograd import Variable
from engine.gpu import use_gpu, first_device, all_devices, device_description
from engine.parameters import output_path_with_subdir
from engine.util.console.logs import print_debug, print_logs


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


def save_model(model, path):
    """
    save the model parameters on disk
    :param model:
    :param path:
    :return:
    """
    print_debug('Saving model: ' + path)
    if type(model) is torch.nn.DataParallel:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_model(path, model=None, model_params={}):
    """
    load the model parameters from disk
    :param model_params:
    :param model:
    :param path:
    :return:
    """
    model_sd = torch.load(path)
    model = model(**model_params)
    model.load_state_dict(model_sd)
    return model


def load_or_create(model_class, from_scratch=True, model_params={}, p_input=one_input, p_label=one_label):

    # update class to dynamically manage the input
    model_class.forward = decorated_forward(model_class.forward, p_input)

    # Constructing the model
    if from_scratch:
        model = model_class(**model_params)
    else:  # recover from breakpoint
        path = output_path_with_subdir('models', '_model.torch')

        print_logs('Loading model: ' + path)
        model = load_model(path, model_class, model_params=model_params)

    # configure usage on GPU
    if use_gpu():
        model.to(first_device())
        model = torch.nn.DataParallel(model, device_ids=all_devices())

    model.p_label = p_label

    # print info about devices
    print_logs('Device(s)): ' + str(device_description()))

    return model


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
