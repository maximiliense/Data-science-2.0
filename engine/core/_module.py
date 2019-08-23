import time

from engine.path.path import add_config_elements
from engine.logging import print_info, print_durations, format_dict_and_tuple
from engine.parameters import hyper_parameters as hp
from engine.util.merge_dict import merge


def module(func):
    """
    This wrapper encapsulate a function into a module. A module enables dynamic change of the parameters,
    evaluation of execution time, logs, etc.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        print_info('[Executing ' + func.__name__ + ']')

        # check changeable parameters (command line and more)
        if func.__name__ in hp.overriding_parameters():
            for arg, name in zip(args, func.__code__.co_varnames):
                kwargs[name] = arg
            args = tuple()
            merge(kwargs, hp.overriding_parameters()[func.__name__])
        if len(args) > 0 or len(kwargs) > 0:
            add_config_elements('[' + func.__name__ + ']')
        if len(args) > 0:
            print_info('Args: ' + format_dict_and_tuple(args))
            add_config_elements('Args: ' + format_dict_and_tuple(args))

        if len(kwargs) > 0:
            print_info('Kwargs: ' + format_dict_and_tuple(kwargs))
            add_config_elements('Kwargs: ' + format_dict_and_tuple(kwargs))

        results = func(*args, **kwargs)
        print_durations(time.time() - start)
        return results

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__dict__.update(func.__dict__)
    return wrapper
