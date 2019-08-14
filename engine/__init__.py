from engine.machines import detect_machine, check_interactive_cluster
from engine.parameters.ds_argparse import ask_general_config_default
from engine.parameters.path import export_config


def configure_engine():

    import getpass
    import time

    from engine.gpu import set_devices
    from engine.parameters import special_parameters
    from engine.parameters.special_parameters import last_experiment, configure_homex
    from engine.tensorboard import initialize_tensorboard
    from engine.util.clean import clean
    from engine.util.console.logs import print_h1, print_logs, print_durations, print_debug, print_errors
    from engine.util.console.time import get_start_datetime
    from engine.util.console.welcome import print_welcome_message, print_goodbye
    from engine.parameters.ds_argparse import get_argparse, check_general_config, process_other_options
    import sys

    import atexit

    import os

    from engine.parameters.hyper_parameters import check_parameters, check_config, _overriding_parameters

    args = get_argparse()

    # def warn(*a, **k):
    #     pass
    import warnings
    # warnings.warn = warn
    warnings.filterwarnings('default', category=DeprecationWarning)

    process_other_options(args.more)

    # general setup
    special_parameters.verbose = args.verbose
    special_parameters.debug = args.debug
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

    # hardware
    set_devices(args.gpu)
    special_parameters.nb_workers = args.nb_workers
    special_parameters.nb_nodes = args.nb_nodes

    special_parameters.first_epoch = args.epoch
    special_parameters.validation_id = args.validation_id

    load_model = (args.epoch != 1 or args.validation_only or args.export)
    special_parameters.from_scratch = args.from_scratch if args.from_scratch is not None else not load_model

    config_name = check_config(args)
    check_parameters(args)
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

    if not args.serious:
        print_welcome_message()

    # tensorboard
    if special_parameters.tensorboard:
        initialize_tensorboard()

    start_dt = get_start_datetime()

    print_logs('Starting datetime: ' + start_dt.strftime('%Y-%m-%d %H:%M:%S'))
    if not special_parameters.from_scratch:
        _, special_parameters.experiment_name = last_experiment(special_parameters.output_name)
        # special_parameters.output_name = name
        if special_parameters.experiment_name is None:
            print_errors('No previous experiment named ' + special_parameters.output_name, do_exit=True)
    else:
        special_parameters.experiment_name = special_parameters.output_name + '_' + start_dt.strftime('%Y%m%d%H%M%S')

    export_config()
    atexit.register(exit_handler)


configure_engine()
