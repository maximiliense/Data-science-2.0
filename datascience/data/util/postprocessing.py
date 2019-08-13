from sklearn.neighbors import KDTree
from engine.util.logs import print_debug
import numpy as np


class PostProcessing:
    pass


class AddTrainKNNIndex(PostProcessing):
    def __init__(self, leaf_size=2):
        self.leaf_size = leaf_size

    def __call__(self, train, validation, test):
        # TODO save knn if test's size < 1 and load knn if test's size is 1...
        # use output_path to generate the path when saving and output_path_with_validation to generate
        # the path when loading. _with_validation will check if the knn is saved under a different name
        print_debug("Constructing spatial knn index...")
        train_dataset = train.dataset
        kdtree = KDTree(train_dataset, leaf_size=self.leaf_size)
        train.kdtree = kdtree
        validation.kdtree = kdtree
        test.kdtree = kdtree
        train.train_dataset = train
        validation.train_dataset = train
        test.train_dataset = train


class ExtractCooccurrencesMultipoints(PostProcessing):
    def __init__(self, leaf_size=2, dataset_to_use=None):
        self.dataset = dataset_to_use
        self.leaf_size = leaf_size

    def __call__(self, train, validation, test, nb_labels=3336):
        print_debug("Extract cooccurrences...")
        if self.dataset is None:
            self.dataset = train

        train_dataset = self.dataset.dataset

        multipoints = {}
        for i, pos in enumerate(train_dataset):
            if pos not in multipoints:
                multipoints[pos] = np.zeros(nb_labels, dtype=int)
            multipoints[pos][self.dataset.labels[i]] += 1
        kdtree = KDTree(list(multipoints.keys()), leaf_size=self.leaf_size)

        train.pos_multipoints = list(multipoints.keys())
        validation.pos_multipoints = list(multipoints.keys())
        test.pos_multipoints = list(multipoints.keys())

        train.kdtree = kdtree
        validation.kdtree = kdtree
        test.kdtree = kdtree

        train.multipoints = multipoints
        validation.multipoints = multipoints
        test.multipoints = multipoints
