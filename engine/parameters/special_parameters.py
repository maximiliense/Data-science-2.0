"""
contains all special parameters
"""
import os

import json
import sys

from engine.flags import deprecated

root_path = None
project_path = None
homex = ''
source_path = 'sources/'
setup_name = ''
output_name = ''
experiment_name = None

validation_id = None

train = False
evaluate = False
export = False

first_epoch = 0

load_model = False
restart_experiment = False

machine = 'auto'

# if interactive cluster is False, then the code should not ask and wait for user's input
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
plt_default_size = (18, 14)

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


def subexperiment_name(name):
    """
    for doing multiple experiment in the same execution
    :param name:
    :return:
    """
    global xp_name
    xp_name = name


def configure(args):
    if args.general_config is not None:
        if not args.general_config.endswith("json"):
            args.general_config = 'configs/' + args.general_config + '.json'

        with open(args.general_config) as f:
            d = json.load(f)

        for k, v in d.items():
            globals()[k] = v
