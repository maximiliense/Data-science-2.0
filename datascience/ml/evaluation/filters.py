import numpy as np

from datascience.data.util.index import reverse_indexing, get_index
from engine.path import output_path


class Filter(object):
    pass


class FilterLabelsList(Filter):
    def __init__(self, filter_file_path):
        self.filter_file_path = filter_file_path
        self.filter = None

    def __call__(self, input_data):
        if self.filter is None:
            self._config(input_data)

        np.multiply(input_data, self.filter, out=input_data)

    def _config(self, input_data):
        self.filter = np.zeros((input_data.shape[1],))
        index_path = output_path('index.json')
        indexed_labels = reverse_indexing(get_index(index_path))

        with open(self.filter_file_path) as f:
            for l in f:
                if int(l) in indexed_labels:
                    self.filter[indexed_labels[int(l)]] = 1.
                elif indexed_labels is None:
                    self.filter[int(l)] = 1.
