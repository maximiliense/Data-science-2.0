# from engine.util import special_parameters
import sys

from torch import nn
from torch.nn import DataParallel

from engine.logging.verbosity import is_debug, is_verbose
from engine.util.console.print_colors import color
from inspect import isclass, getmodule


def print_info(log, end='\n'):
    """
    normal print
    :param end:
    :param log:
    :return:
    """
    print(log, end=end)


def print_statistics(log, end='\n'):
    """

    :param log:
    :param end:
    :return:
    """
    print(color.YELLOW + log + color.END, end=end)


def print_notif(log, end='\n'):
    """

    :param log:
    :param end:
    :return:
    """
    print(color.BLUE + log + color.END, end=end)


def print_h2(log, end='\n'):
    """
    print header
    :param end:
    :param log:
    :return:
    """
    print(color.GREEN + log + color.END, end=end)


def print_h1(log, end='\n'):
    """
    :param log:
    :param end:
    :return:
    """
    print(color.GREEN + '\n' + '*' * 10 + ' ' + log + ' ' + '*' * 10 + '\n' + color.END, end=end)


def print_logs(log, end='\n'):
    """
    print logs if verbose
    :param end:
    :param log:
    :return:
    """
    if is_debug() or is_verbose():
        print(color.RED + log + color.END, end=end)
        sys.stdout.flush()


def print_durations(duration, text='duration', end='\n'):

    h = str(int(duration // 3600)).zfill(2)
    duration %= 3600
    m = str(int(duration // 60)).zfill(2)
    duration %= 60
    s = str(int(duration // 1)).zfill(2)
    print_logs('[' + text + ': %s:%s:%s]' % (h, m, s), end=end)


def print_debug(log, end='\n'):
    """
    print logs if verbose
    :param end:
    :param log:
    :return:
    """
    if is_debug():
        print(color.RED + log + color.END, end=end)
        sys.stdout.flush()


def print_debug_verbose(log, end='\n'):
    """
    print logs if verbose
    :param end:
    :param log:
    :return:
    """
    if is_verbose() and is_debug():
        print(color.RED + log + color.END, end=end)
        sys.stdout.flush()


def print_errors(log, end='\n', do_exit=False):
    """

    :param do_exit:
    :param log:
    :param end:
    :return:
    """
    print(color.RED + '\t*** ' + log + ' ***' + color.END, end=end)
    if do_exit:
        exit()


def print_dataset_statistics(train_size, validation_size, test_size, source_name, labels_size):
    """
    print math about the loaded data
    :param train_size:
    :param validation_size:
    :param test_size:
    :param source_name:
    :param labels_size:
    """
    source_name_str = ' source name : ' + source_name + ' '
    nb_dash_1 = max(0, 50 - len(source_name_str)) // 2
    nb_dash_2 = max(0, 50 - len(source_name_str) - nb_dash_1)
    print_statistics('-' * nb_dash_1 + source_name_str + '-' * nb_dash_2)
    if train_size > 0:
        print_statistics('Train size: ' + str(train_size))
    if validation_size > 0:
        print_statistics('Validation size: ' + str(validation_size))
    if test_size > 0:
        print_statistics('Test size: ' + str(test_size))

    print_statistics('Labels indexed: ' + str(labels_size))
    print_statistics('-' * 50)


def format_dict_and_tuple(d, t=0, tab='  '):

    result = '{\n' if type(d) is dict else '(\n'
    result += _format_dict(d, t, tab) if type(d) is dict else _format_tuple(d, t, tab)
    result += tab * t + '}' if type(d) is dict else tab * t + ')'
    return result


def _format_dict(d, t=0, tab='  '):
    result = ''
    for k, v in d.items():
        if type(v) is not dict and type(v) is not tuple:
            result += tab * (t+1) + str(k) + ': ' + _object_str(v) + ',\n'
        else:
            result += tab * (t+1) + str(k) + ': ' + format_dict_and_tuple(v, t + 1) + ',\n'
    return result


def _format_tuple(d, t=0, tab='  '):
    result = ''
    for v in d:
        if type(v) is not dict and type(v) is not tuple:
            result += tab * (t + 1) + _object_str(v) + ',\n'
        else:
            result += tab * (t + 1) + format_dict_and_tuple(v, t + 1) + ',\n'
    return result


def _object_str(el):

    if isclass(el):
        return el.__name__
    elif isinstance(el, nn.Module):
        if type(el) is DataParallel:
            el = el.module
        repr_filename = el.__repr__.__code__.co_filename
        base_repr_filename = getmodule(el.__class__.__base__).__file__
        if repr_filename == base_repr_filename and (not is_verbose() or not is_debug()):
            return 'model(' + el.__class__.__name__ + ')'
        else:
            return repr(el)
    else:
        return repr(el)
