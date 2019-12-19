from engine.logging import print_errors


def merge(dict_a, dict_b):
    """
    Merge dict_b into dict_a recursively
    :param dict_a:
    :param dict_b:
    """
    for k in dict_b.keys():
        if k in dict_a and type(dict_a[k]) is dict and type(dict_b[k]) is dict:
            merge(dict_a[k], dict_b[k])
        else:
            dict_a[k] = dict_b[k]


def merge_smooth(dict_a, dict_b):
    """
    Merge dict_b into dict_a recursively. keep existing keeps in dict_a
    :param dict_a:
    :param dict_b:
    """
    for k in dict_b.keys():
        if k in dict_a and type(dict_a[k]) is dict and type(dict_b[k]) is dict:
            merge(dict_a[k], dict_b[k])
        elif k not in dict_a:
            dict_a[k] = dict_b[k]


def merge_dict_set(*args):
    """
    args contains a series of dictionaries each followed by the default value. merge_dict_set, will set
    the default values if not already set
    :param args:
    :return:
    """
    if len(args) % 2 != 0:
        print_errors('args must contains a multiple of 2 elements', exception=MergeDictException('multiple of 2'))

    results = []

    for i in range(0, len(args), 2):
        # the currently set parameters
        dictionary = args[i] if args[i] is not None else {}
        if dictionary is not None and type(dictionary) is not dict:
            print_errors('arguments should be either None or a dict', exception=MergeDictException('dict or None'))

        # default values of the parameters
        default = args[i + 1]
        if type(default) is not dict:
            print_errors('default values should be of type dict', exception=MergeDictException('dict'))

        merge_smooth(
            dictionary,
            default
        )
        results.append(dictionary)
    return results


class MergeDictException(Exception):
    def __init__(self, message):
        super(MergeDictException, self).__init__(message)
