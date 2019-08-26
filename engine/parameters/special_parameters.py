"""
contains all special parameters
"""
import os

import json
import re
import sys

from engine.flags import deprecated

root_path = None
project_path = None
homex = None
source_path = 'sources/'
setup_name = ''
output_name = ''
experiment_name = None

validation_id = None

validation_only = False
export = False

first_epoch = 0

from_scratch = True

machine = 'auto'

interactive_cluster = True

tensorboard_path = '/home/data/runs'  # tensorboard data will be saved here
tensorboard_writer = None
tensorboard = False

nb_workers = 0
nb_nodes = 1

check_known_parameters = True

file = False
mail = 0
to_mail = ''
to_name = 'John Smith'

plt_style = 'dark_background'

other_options = []

xp_name = ''


def get_validation_id():
    global validation_id
    return validation_id


def get_parameters(param_name, default_value):
    """
    return parameters that available in this module... Notice that these parameters can be
    dynamically set by command line
    :param param_name: the name of the parameter
    :param default_value:  its default value if not available
    :return the param value if available or default value
    """
    return getattr(sys.modules[__name__], param_name, default_value)


def configure_homex():
    global homex
    if homex is None:
        homex = os.path.join(root_path, project_path, setup_name)
    else:
        homex = os.path.join(homex, setup_name)
        if not os.path.exists(homex):
            os.mkdir(homex)


def subexperiment_name(name):
    """
    for doing multiple experiment in the same execution
    :param name:
    :return:
    """
    global xp_name
    xp_name = name


@deprecated(comment='Should use engine.parameters.path.output_path')
def output_path(ext):
    create_xp_dir()
    iter_id = '' if xp_name == '' else '_' + xp_name
    return os.path.join(
        homex, experiment_name, output_name + iter_id + ext
    )


@deprecated(comment='Should use engine.parameters.path.output_path_without_validation')
def output_path_without_validation(ext):
    path = output_path(ext)
    path = path if os.path.isfile(path) else re.sub(r"_[0-9]+" + ext, ext, path)
    return path


@deprecated(comment='Should not use anymore')
def create_xp_dir():
    dir_path = os.path.join(homex, experiment_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


@deprecated(comment='Should not use anymore')
def create_dir(directory):
    create_xp_dir()
    dir_path = os.path.join(homex, experiment_name, directory)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


@deprecated(comment='Should use engine.parameters.path.output_path')
def output_path_with_subdir(directory, ext):
    path = create_dir(directory)
    iter_id = '' if xp_name == '' else '_' + xp_name
    return os.path.join(path, output_name + iter_id + ext)


def configure(args):
    if args.general_config is not None:
        if not args.general_config.endswith("json"):
            args.general_config = 'configs/' + args.general_config + '.json'

        with open(args.general_config) as f:
            d = json.load(f)

        for k, v in d.items():
            globals()[k] = v


def last_experiment(name):
    last = _last_experiment(name)
    if last is None:
        regex = '^(.*)_[0-9]+$'
        name = re.sub(regex, r'\1', name)
        last = _last_experiment(name)
    return name, last


def _last_experiment(name):
    experiment = None
    timestamp = 0
    for l in os.listdir(homex):
        regex = '^' + name + '_20[0-9]{10,13}$'
        if os.path.isdir(os.path.join(homex, l)) and re.match(regex, l) is not None:
            t = most_recent_file(os.path.join(homex, l))
            if experiment is None or t > timestamp:
                experiment = l
                timestamp = t
    return experiment


def most_recent_file(path):
    f = max([os.path.join(path, l) for l in os.listdir(path)], key=os.path.getctime)
    return os.path.getctime(f)
