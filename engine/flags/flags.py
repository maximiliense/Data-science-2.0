import warnings


def deprecated(comment=''):
    comment = '' if comment == '' else ' (' + comment + ')'

    def _deprecated(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""

        def new_func(*args, **kwargs):
            what = 'function' if type(func) is not type else 'class'
            warnings.warn(
                'Call to deprecated %s %s%s.' % (what, func.__name__, comment), category=DeprecationWarning
            )
            return func(*args, **kwargs)

        set_func_attributes(new_func, func)
        return new_func

    return _deprecated


def duplicated(func):

    def wrapper(*args, **kwargs):
        warnings.warn('Code is duplicated.', category=DeprecationWarning)
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__dict__.update(func.__dict__)
    return wrapper


def incorrect_structure(details):
    def _incorrect_structure(func):
        def new_func(*args, **kwargs):
            warnings.warn('Incorrect code structure for %s: %s.' % (func.__name__, details), category=Warning)
            return func(*args, **kwargs)

        set_func_attributes(new_func, func)
        return new_func

    return _incorrect_structure


def gpu_cpu(details=''):
    details = '' if details == '' else ' (' + details + ')'

    def _gpu_cpu(func):
        def new_func(*args, **kwargs):
            warnings.warn('Incorrect GPU/CPU support for %s%s.' % (func.__name__, details), category=Warning)
            return func(*args, **kwargs)

        set_func_attributes(new_func, func)
        return new_func

    return _gpu_cpu


def incorrect_io(explanation=''):
    explanation = '' if explanation == '' else ' (' + explanation + ')'

    def _incorrect_io(func):
        def new_func(*args, **kwargs):
            warnings.warn('Incorrect IO support for %s%s.' % (func.__name__, explanation), category=Warning)
            return func(*args, **kwargs)

        set_func_attributes(new_func, func)
        return new_func

    return _incorrect_io


def set_func_attributes(new_func, old_func):
    new_func.__name__ = old_func.__name__
    new_func.__doc__ = old_func.__doc__
    new_func.__dict__.update(old_func.__dict__)
