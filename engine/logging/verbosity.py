verbose = False
debug = False


def is_debug():
    global debug
    return debug


def is_verbose():
    global verbose
    return verbose


def set_debug(deb):
    global debug
    debug = deb


def set_verbose(verb):
    global verbose
    verbose = verb
