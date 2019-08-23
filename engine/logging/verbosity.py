_verbose = 0


def is_debug():
    global _verbose
    return _verbose >= 3


def verbose_level():
    global _verbose
    return _verbose


def is_info():
    global _verbose
    return _verbose >= 2


def is_warning():
    global _verbose
    return _verbose >= 1


def set_verbose(verb):
    global _verbose
    _verbose = verb
