import random
from math import ceil
import numpy as np
from sklearn.utils import indexable, shuffle

from engine.flags import deprecated


def train_test_split_stratified(*columns, test_size, random_state=42):
    columns = indexable(*columns)

    # convention : labels is the first columns
    labels = columns[0]

    dict_esp = {}
    train_ids = []
    test_ids = []

    # indexing the occurrences by species
    for i, l in enumerate(labels):
        if l not in dict_esp:
            dict_esp[l] = []
        dict_esp[l].append(i)

    # stratified sampling (at least one in train if 1 occurrence or 1 in train and 1 in test if 2 occurrences and more)
    for j, label in enumerate(dict_esp):
        idx = dict_esp[label]
        shuffle(idx, random_state=random_state + j)
        train_ids.append(idx.pop(0))
        sample_size = ceil(test_size * len(idx))
        test_ids.extend(idx[:sample_size])
        train_ids.extend(idx[sample_size:])

    # creating output
    r = []
    for col in columns:
        r.extend((col[train_ids], col[test_ids]))

    return r


class DatasetSplitter:
    pass


@deprecated()
class DatasetSplitterStratifiedOld(DatasetSplitter):
    def __init__(self):
        self.dict_esp = {}
        self.train_ids = []
        self.test_ids = []

    def __call__(self, *columns, test_size, random_state=42):
        dataset = np.asarray(columns[1])
        ids = np.asarray(columns[2])
        labels = np.asarray(columns[0])

        for i, l in enumerate(labels):
            if l not in self.dict_esp:
                self.dict_esp[l] = []
            self.dict_esp[l].append(i)

        for label in self.dict_esp:
            idx = self.dict_esp[label]
            random.shuffle(idx)
            self.train_ids.append(idx.pop(0))
            sample_size = ceil(test_size * len(idx))
            self.test_ids.extend(idx[:sample_size])
            self.train_ids.extend(idx[sample_size:])

        X_tr = dataset[self.train_ids]
        X_te = dataset[self.test_ids]
        y_tr = labels[self.train_ids]
        y_te = labels[self.test_ids]
        id_tr = ids[self.train_ids]
        id_te = ids[self.test_ids]

        return X_tr, X_te, id_tr, id_te, y_tr, y_te
