import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

from engine.logging.logs import print_notif, print_debug

from engine.parameters import special_parameters


def get_argparse():
    """
    Construct the argparse for command line use
    :return:
    """
    parser = argparse.ArgumentParser(description='Data science platform.', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--name', dest='output_name', type=str, default='*')

    group = parser.add_argument_group('Experiment')
    group.add_argument('--epoch', dest='epoch', type=int, default=1)
    group.add_argument('--from-scratch', dest='from_scratch', type=bool, default=None)
    group.add_argument('--validation-id', dest='validation_id', type=int, default=None)
    # export
    group.add_argument('--export', dest='export', action='store_true', default=False)
    group.add_argument('--validation', dest='validation_only', action='store_true', default=False)
    # parameters
    group.add_argument('-p', '--params', dest='params', type=str, default='')
    group.add_argument('-c', '--config', nargs='+', dest='config', default=None, help='list of configs')

    group.add_argument('--clean', dest='clean', action='store_true', default=False, help='Clean tmp files.')
    group.add_argument('--show', dest='show', action='store_true', default=False, help='Show tmp files.')

    group = parser.add_argument_group('Hardware')
    group.add_argument('-g', '--gpu', dest='gpu', type=str, default=None)
    group.add_argument('--workers', dest='nb_workers', type=int, default=16)
    group.add_argument('--nodes', dest='nb_nodes', type=int, default=1)

    group = parser.add_argument_group('Specials')
    group.add_argument('-s', '--serious', dest='serious', action='store_true', default=False)
    group.add_argument('--general-config', dest='general_config', type=str, default=None)
    group.add_argument('--homex', dest='homex', type=str, default=None, help='where the data will be exported.')
    group.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False)
    group.add_argument('-d', '--debug', dest='debug', action='store_true', default=False)
    group.add_argument('--style', dest='style', type=str, default='dark_background')
    group.add_argument('-m', '--more', nargs='+', help='Additional attributes for special_parameters', required=False)

    return parser.parse_args()


def check_general_config(param_list):
    """
    check default config or save current as default
    :param param_list:
    :return: False if config was not set True otherwise
    """
    import os
    if param_list.general_config is None:
        list_rood_dir = os.listdir('.')
        for c in os.listdir('configs'):
            c_name = c.replace('.json', '')
            if c_name in list_rood_dir:

                print_debug('[Using config: ' + c_name + ']')
                param_list.general_config = c_name
        return False
    else:
        return True


def ask_general_config_default(param_list):
    if special_parameters.interactive_cluster:
        a = None
        while a not in ('', 'y', 'Y', 'n', 'N'):
            print_notif('Make config ' + param_list.general_config + ' as default: [Y/n]? ', end='')
            a = input()
        if a in ('', 'y', 'Y'):
            config_name = param_list.general_config.replace('.json', '').replace('configs/', '')
            Path(config_name).touch()
            print_debug('[Config ' + config_name + ' now default]')


def process_other_options(args):
    if args is not None:
        mem = {}
        for option in args:
            exec(option, {}, mem)
        for k, v in mem.items():
            setattr(special_parameters, k, v)
