import time

from engine.path.path import add_config_elements
from engine.logging import print_logs, print_durations, format_dict
from engine.parameters import overriding_parameters
from engine.util.merge_dict import merge


def module(func):
    """
    This wrapper encapsulate a function into a module. A module enables dynamic change of the parameters,
    evaluation of execution time, logs, etc.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        print_logs('[Executing ' + func.__name__ + ']')

        # check changeable parameters (command line and more)
        if func.__name__ in overriding_parameters():
            for arg, name in zip(args, func.__code__.co_varnames):
                kwargs[name] = arg
            args = tuple()
            merge(kwargs, overriding_parameters()[func.__name__])
        if len(args) > 0 or len(kwargs) > 0:
            add_config_elements('[' + func.__name__ + ']')
        if len(args) > 0:
            print_logs('Args: '+str(args))
            add_config_elements('Args: '+str(args))

        if len(kwargs) > 0:
            print_logs('Kwargs: ' + format_dict(kwargs))
            add_config_elements('Kwargs: ' + format_dict(kwargs))

        results = func(*args, **kwargs)
        print_durations(time.time() - start)
        return results

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__dict__.update(func.__dict__)
    return wrapper
