import torch


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
