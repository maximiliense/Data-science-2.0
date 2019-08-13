

class DatasetFilter:
    pass


def online_filters_processing(filters, x):
    for f in filters:
        if f(x):
            return True
    return False


def index_labels(indexer, label):
    # multitask
    if type(label) in (tuple, list):
        for i, l in enumerate(label):
            if l not in indexer[i]:
                indexer[i][l] = len(indexer[i])
        return (indexer[i][l] for i, l in enumerate(label))
    else:
        if label not in indexer:
            indexer[label] = len(indexer)
        return indexer[label]


def is_label_indexed(indexer, label):
    if type(label) is tuple:
        is_indexed = True
        for i, l in enumerate(label):
            is_indexed = is_indexed and l in indexer[i]
        return is_indexed
    else:
        return label in indexer


def filter_test(labels, label_test, x, ids, *args):
    """
    Remove occurrences from test set if they are not already in train set
    :param x:
    :param ids:
    :param labels: label that should be kept in test set
    :param label_test: current test set labels
    :param args: other test set lists to modify according to label_test
    """
    if len(label_test) > 0:
        tmp_indexer = {}  # if type(labels[0]) is not list else [{} for _ in labels[0]]
        for label_list in labels:
            for label in label_list:
                index_labels(tmp_indexer, label)

        selector = [is_label_indexed(tmp_indexer, l) for l in label_test]
        label_test = label_test[selector]
        x = x[selector]
        ids = ids[selector]

        return (label_test, x, ids) + tuple(a[selector] for a in args)
    else:
        return (label_test, x, ids) + args


"""
online filters
"""


class FilterThreshold(DatasetFilter):
    """
    Filter elements given a threshold
    """
    def __init__(self, threshold, column, gt=True):
        self.threshold = threshold
        self.column = column
        self.gt = gt

    def __call__(self, x):
        return (x[self.column] < self.threshold and self.gt) or (x[self.column] > self.threshold and not self.gt)


class FilterLabels(DatasetFilter):
    """
    Filter elements that are in a list
    """
    def __init__(self, labels, column):
        self.labels = labels
        self.column = column

    def __call__(self, x):
        return x[self.column] in self.labels
