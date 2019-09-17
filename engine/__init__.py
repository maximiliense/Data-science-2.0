from engine.logging import print_debug, is_warning
from engine.machines import detect_machine, check_interactive_cluster
from engine.parameters.ds_argparse import ask_general_config_default
from engine.parameters.hyper_parameters import list_aliases
from engine.path.path import export_config, output_directory, load_last_epoch
from engine.logging.verbosity import set_verbose, is_info


def configure_engine():

    import getpass
    import time

    from engine.hardware import set_devices
    from engine.parameters import special_parameters
    from engine.parameters.special_parameters import last_experiment, configure_homex
    from engine.tensorboard import initialize_tensorboard
    from engine.util.clean import clean
    from engine.logging.logs import print_h1, print_info, print_durations, print_info, print_errors, print_info
    from engine.util.console.time import get_start_datetime
    from engine.util.console.welcome import print_welcome_message, print_goodbye
    from engine.parameters.ds_argparse import get_argparse, check_general_config, process_other_options
    import sys

    import atexit

    import os

    from engine.parameters import hyper_parameters as hp

    args = get_argparse()

    # def warn(*a, **k):
    #     pass
    import warnings
    # warnings.warn = warn
    warnings.filterwarnings('default', category=DeprecationWarning)

    process_other_options(args.more)

    # general setup
    set_verbose(args.verbose)
    special_parameters.plt_style = args.style
    special_parameters.homex = args.homex

    special_parameters.setup_name = os.path.split(os.path.split(sys.argv[0])[0])[1]
    special_parameters.project_path = os.path.split(os.path.split(sys.argv[0])[0])[0]

    ask_default = check_general_config(args)

    special_parameters.configure(args)

    if special_parameters.machine == 'auto':
        detect_machine()

    special_parameters.interactive_cluster = check_interactive_cluster(special_parameters.machine)

    if ask_default:
        ask_general_config_default(args)

    special_parameters.root_path = os.path.abspath(os.curdir)

    configure_homex()

    if args.clean or args.show:
        clean(special_parameters.homex, args.output_name, disp_only=args.show)
        exit()
    if args.list_aliases:
        list_aliases()
        exit()

    # hardware
    set_devices(args.gpu)
    special_parameters.nb_workers = args.nb_workers
    special_parameters.nb_nodes = args.nb_nodes

    special_parameters.first_epoch = args.epoch - 1  # to be user friendly, we start at 1
    special_parameters.validation_id = args.validation_id

    config_name = hp.check_config(args)
    hp.check_parameters(args)
    default_name = os.path.split(sys.argv[0])[-1].replace('.py', '')
    if args.output_name == '*':
        if config_name is None:
            special_parameters.output_name = default_name
        else:
            c_name = config_name.split('/')[-1]
            special_parameters.output_name = default_name + '_' + c_name
    else:
        special_parameters.output_name = args.output_name

    def exit_handler():
        if special_parameters.tensorboard_writer is not None:
            special_parameters.tensorboard_writer.close()

        print_durations(time.time() - start_dt.timestamp(), text='Total duration')

        if args.serious:
            print_h1("Goodbye")
        else:
            print_goodbye()

    print_h1('Hello ' + getpass.getuser() + '!')

    if not args.serious and is_warning():
        print_welcome_message()

    start_dt = get_start_datetime()

    print_info('Starting datetime: ' + start_dt.strftime('%Y-%m-%d %H:%M:%S'))

    # configuring experiment
    load_model = (args.epoch != 1 or args.validation_only or args.export or args.restart)
    special_parameters.from_scratch = not args.load_model if args.load_model is not None else not load_model

    if not special_parameters.from_scratch:
        _, special_parameters.experiment_name = last_experiment(special_parameters.output_name)
        # special_parameters.output_name = name
        if special_parameters.experiment_name is None:
            print_errors('No previous experiment named ' + special_parameters.output_name, do_exit=True)

        if args.restart:
            special_parameters.first_epoch = load_last_epoch()
            print_debug('Restarting experiment at last epoch: {}'.format(special_parameters.first_epoch))
    else:
        special_parameters.experiment_name = special_parameters.output_name + '_' + start_dt.strftime('%Y%m%d%H%M%S')

    if not is_info():
        print('Output directory: ' + output_directory() + '\n')

    # tensorboard
    if special_parameters.tensorboard:
        initialize_tensorboard()

    export_config()
    atexit.register(exit_handler)


configure_engine()
