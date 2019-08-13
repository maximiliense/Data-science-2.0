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
