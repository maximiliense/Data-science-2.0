import os
import re
from engine.parameters import special_parameters


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
