from engine.logging.logs import print_errors


def perform_split(dataset, test_size, splitter, random_state=42, **kwargs):
    """
    Perform a split even if len(data) == 0 or test_size in (0,1)
    :param dataset:
    :param test_size:
    :param splitter:
    :param random_state
    :param quadra_size
    :return:
    """
    quad_size = kwargs['quad_size']
    if len(dataset) == 0:
        print_errors('[perform_split] data should be a list, a tuple, etc. of positive size', do_exit=True)
    print(quad_size)
    if 0 < test_size < 1 and len(dataset[0]) > 0:
        split_test = splitter(*dataset, test_size=test_size, random_state=random_state, quad_size=quad_size)
        train, test = split_test[::2], split_test[1::2]
    elif test_size == 0:
        train, test = [c for c in dataset], [[] for _ in dataset]
    else:
        train, test = [[] for _ in dataset], [c for c in dataset]
    return train, test
