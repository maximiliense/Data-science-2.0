import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

from engine.logging.logs import print_notification, print_info

from engine.parameters import special_parameters

"""
If you want to add an option in argparse, please submit an issue: 
https://github.com/maximiliense/Data-science-2.0/issues and attach it to the **engine** project.
"""


def get_argparse():
    """
    Construct the argparse for command line use
    :return: argparse object
    """
    parser = argparse.ArgumentParser(description='Data science platform.', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-n', '--name', dest='output_name', type=str, default='*',
                        help='Set the name of the current execution. Otherwise generated automatically.')

    group = parser.add_argument_group('Experiment')

    group.add_argument('-e', '--epoch', dest='epoch', type=int, default=1,
                       help='Starts the training at the given epoch (default: 1)')

    group.add_argument('-l', '--load-model', dest='load_model', action='store_true', default=None,
                       help='Load model from previous execution (default: Auto).')

    group.add_argument('-i', '--validation-id', dest='validation_id', type=int, default=None,
                       help='when loading a specific model, use the one used at a specific validation (default: None)')

    group.add_argument('-x', '--export', dest='export', action='store_true', default=False,
                       help='At the end of the training (if training there is), export the results in a file ('
                            'default: False)')

    group.add_argument('-vo', '--validation', dest='validation_only', action='store_true', default=False,
                       help='Do not train the model and execute a validation only (default: False)')

    group.add_argument('-p', '--params', dest='params', type=str, default='',
                       help='Parameters to override module calls (default=\'\')')

    group.add_argument('-c', '--config', nargs='+', dest='config', default=None,
                       help='Use configs to override module calls. Multiple configs can be given (default: None)')

    group.add_argument('--clean', dest='clean', action='store_true', default=False,
                       help='Clean generated files (default: False).')

    group.add_argument('--show', dest='show', action='store_true', default=False,
                       help='Show generated files (default: False).')

    group = parser.add_argument_group('Hardware')
    group.add_argument('-g', '--gpu', dest='gpu', type=str, default=None, help='Ask for GPUs (default: None) ')

    group.add_argument('-w', '--workers', dest='nb_workers', type=int, default=16,
                       help='Change the default number of parallel workers (default: 16)')

    group.add_argument('-no', '--nodes', dest='nb_nodes', type=int, default=1,
                       help='Ask for multiple nodes on a cluster (default: 1)')

    group = parser.add_argument_group('Specials')

    group.add_argument('-s', '--serious', dest='serious', action='store_true', default=False,
                       help='Hide the funny messaged displayed at the beginning and at the end of an execution '
                            '(default: False)')

    group.add_argument('-gc', '--general-config', dest='general_config', type=str, default=None,
                       help='Load a configuration file to adapt the framework execution (default: None)')

    group.add_argument('--homex', dest='homex', type=str, default=None,
                       help='Change the destination of the exported files. By default the data are exported in the '
                            'experiment folder (default:None).')

    group.add_argument('-v', '--verbose', dest='verbose', action='store', default=0, const=2, nargs='?', type=int,
                       help='Verbosity (0: silent, 1: warnings, 2: info, 3: debug')

    group.add_argument('--style', dest='style', type=str, default='dark_background',
                       help='Change the plots style (default: dark_background)')
    group.add_argument('-m', '--more', nargs='+', help='Additional attributes for special_parameters', required=False)

    # group.add_argument('--mode', dest='mode', type=str, default=None)

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

                print_info('[Using config: ' + c_name + ']')
                param_list.general_config = c_name
        return False
    else:
        return True


def ask_general_config_default(param_list):
    if special_parameters.interactive_cluster:
        a = None
        while a not in ('', 'y', 'Y', 'n', 'N'):
            print_notification('Make config ' + param_list.general_config + ' as default: [Y/n]? ', end='')
            a = input()
        if a in ('', 'y', 'Y'):
            config_name = param_list.general_config.replace('.json', '').replace('configs/', '')
            Path(config_name).touch()
            print_info('[Config ' + config_name + ' now default]')


def process_other_options(args):
    if args is not None:
        mem = {}
        for option in args:
            exec(option, {}, mem)
        for k, v in mem.items():
            setattr(special_parameters, k, v)
