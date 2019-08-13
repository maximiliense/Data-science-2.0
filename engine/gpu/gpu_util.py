import torch
import ast

devices = None


def nb_gpu():
    """
    :return: number of GPUs that have been asked
    """
    global devices
    if devices is not None:
        return len(devices)
    else:
        return 0


def use_gpu():
    """
    :return: True if the code runs on GPU False otherwise
    """
    global devices
    return devices is not None and 'cuda' in str(devices[0])


def set_devices(args):
    """
    set the devices to use
    :param args: None if CPU the list of GPUs otherwise
    :return:
    """
    global devices
    if args is not None:
        devices = [torch.device(i) for i in ast.literal_eval('[' + args + ']')]
        torch.cuda.set_device(devices[0])
    else:
        devices = [torch.device('cpu')]


def first_device():
    """
    :return: the first device that has been set, or None otherwise
    """
    global devices
    if devices is not None:
        return devices[0]


def all_devices():
    """
    :return: the list of devices if set or None otherwise
    """
    global devices
    return devices


def device_description():
    """
    :return: a string that describe the devices that are used
    """
    global devices
    return [str(g) for g in devices]
