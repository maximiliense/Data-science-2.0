from sklearn.neighbors import KDTree
from engine.logging.logs import print_debug
import numpy as np


def extract_cooccurrences_multipoints(train, test, leaf_size=2, validation=None, nb_labels=3336):
    print_debug("Extract cooccurrences...")
    train_dataset = train.dataset
    multipoints = {}
    for i, pos in enumerate(train_dataset):
        if pos not in multipoints:
            multipoints[pos] = np.zeros(nb_labels, dtype=int)
        multipoints[pos][train.labels[i]] += 1
    kdtree = KDTree(list(multipoints.keys()), leaf_size=leaf_size)

    train.pos_multipoints = list(multipoints.keys())
    test.pos_multipoints = list(multipoints.keys())

    train.kdtree = kdtree
    test.kdtree = kdtree

    train.multipoints = multipoints
    test.multipoints = multipoints

    if validation is not None:
        validation.pos_multipoints = list(multipoints.keys())
        validation.kdtree = kdtree
        validation.multipoints = multipoints