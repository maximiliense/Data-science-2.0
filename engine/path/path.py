import os
import sys

from engine.logging import print_info
import re
from engine.parameters import special_parameters


def output_directory():
    return os.path.join(special_parameters.homex, special_parameters.experiment_name)


def output_path(filename, validation_id=None, have_validation=False):
    """
    return the path given the requested file name. Construct the hierarchy if necessary.
    :param have_validation: True if the file type can have a validation ID
    :param validation_id: if not None, validation_id will be added to the file
    :param filename: the filename, eventually with some directories e.g. models/model_1.torch
    :return: the path
    """

    if (have_validation and special_parameters.validation_id is not None) or validation_id is not None:
        sp = os.path.splitext(filename)
        validation_id = validation_id if validation_id is not None else special_parameters.validation_id
        filename = '{}_{}{}'.format(sp[0], validation_id, sp[1])

    filename = filename if special_parameters.xp_name == '' else special_parameters.xp_name + '_' + filename

    # construct the hierarchy
    folder_list = _sub_folders_from_path(
        os.path.join(special_parameters.homex, special_parameters.experiment_name, filename)
    )
    current_path = ''
    for i in range(len(folder_list) - 1):
        current_path = os.path.join(current_path, folder_list[i])

        # add directory in the hierarchy if it does not exist yet
        if not os.path.isdir(current_path):
            os.mkdir(current_path)
    # return the file name added to the hierarchy
    return os.path.join(current_path, folder_list[-1])


def list_files(path):
    """
    return a list of files after analysing recursively a folder.
    :param path:
    :return:
    """
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]


def _sub_folders_from_path(path):
    """
    divide a path in its sub folder
    :param path:
    :return:
    """
    folders = []
    while path is not None:
        sp = os.path.split(path)
        if sp[0] != '/':
            folders.append(sp[1])
            path = sp[0]
        else:
            folders.append(sp[0] + sp[1])
            path = None
    folders.reverse()
    return folders


def export_config():
    path = output_path('config.txt')
    print_info('Writing config at: ' + path)
    with open(path, 'a') as f:
        f.write(' '.join(sys.argv) + '\n')


def add_config_elements(element):
    path = output_path('config.txt')
    with open(path, 'a') as f:
        f.write(element + '\n')


def export_epoch(epoch):
    """
    save the last epoch index
    :param epoch:
    :return:
    """
    path = output_path('last_epoch.txt')
    with open(path, 'w') as f:
        f.write(str(epoch))


def load_last_epoch():
    """
    return the last epoch index
    :return:
    """
    path = output_path('last_epoch.txt')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            last_epoch = f.read()
        return int(last_epoch)
    else:
        return 0


def last_experiment_path(name):
    """
    return the path to the last experiment with name 'name'
    :param name:
    :return:
    """
    _, folder_name = last_experiment(name)
    return os.path.join(special_parameters.homex, folder_name)


def last_experiment(name):
    """
    returns the path to the last experiment to have the name set
    :param name:
    :return:
    """
    last = _last_experiment(name)
    if last is None:
        regex = '^(.*)_[0-9]+$'
        name = re.sub(regex, r'\1', name)
        last = _last_experiment(name)
    return name, last


def _last_experiment(name):
    experiment = None
    timestamp = 0
    for l in os.listdir(special_parameters.homex):
        regex = '^' + name + '_20[0-9]{10,13}$'
        if os.path.isdir(os.path.join(special_parameters.homex, l)) and re.match(regex, l) is not None:
            t = most_recent_file(os.path.join(special_parameters.homex, l))
            if experiment is None or t > timestamp:
                experiment = l
                timestamp = t
    return experiment


def most_recent_file(path):
    f = max([os.path.join(path, l) for l in os.listdir(path)], key=os.path.getctime)
    return os.path.getctime(f)


def configure_homex():
    if special_parameters.homex is None:
        special_parameters.homex = os.path.join(special_parameters.root_path,
                                                special_parameters.project_path,
                                                special_parameters.setup_name)
    else:
        special_parameters.homex = os.path.join(special_parameters.homex,
                                                special_parameters.setup_name)
        if not os.path.exists(special_parameters.homex):
            os.mkdir(special_parameters.homex)
