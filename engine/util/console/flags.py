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

        new_func.__name__ = func.__name__
        new_func.__doc__ = func.__doc__
        new_func.__dict__.update(func.__dict__)
        return new_func

    return _deprecated


def incorrect_structure(details):
    details = '' if details == '' else ' (' + details + ')'

    def _incorrect_structure(func):
        def new_func(*args, **kwargs):
            warnings.warn('Incorrect code structure for %s: %s.' % (func.__name__, details), category=Warning)
            return func(*args, **kwargs)

        new_func.__name__ = func.__name__
        new_func.__doc__ = func.__doc__
        new_func.__dict__.update(func.__dict__)
        return new_func

    return _incorrect_structure
