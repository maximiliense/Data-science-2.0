from sklearn.model_selection import train_test_split

from datascience.data.loader.occurrence_loader import _occurrence_loader
from datascience.data.util.source_management import check_source
from engine.core import module
from engine.util.merge_dict import merge_smooth


@module
def occurrence_loader(dataset_class, source=None, validation_size=0.1, test_size=0.1, splitter=train_test_split,
                      filters=tuple(), online_filters=tuple(), postprocessing=tuple(), save_index='default', limit=None,
                      **kwargs):
    """
    Load an occurrence dataset.
    :param dataset_class: the type of dataset (with rasters or not, etc.)
    :param source: the source name
    :param validation_size: [0, 1]
    :param test_size: [0, 1]
    :param splitter: the train test split. By default train_test_split from sklearn
    :param filters: post filters
    :param online_filters: filters that are applied when loading the data
    :param postprocessing: additional transformations
    :param save_index: load, save, default, load_and_save, auto
    :param limit: the number of elements to load
    :param kwargs:
    :return: train, validation, test sets
    """
    if source is not None:
        r = check_source(source)
        merge_smooth(kwargs, r)

    return _occurrence_loader(dataset_class, validation_size=validation_size, test_size=test_size, splitter=splitter,
                              filters=filters, online_filters=online_filters, postprocessing=postprocessing,
                              save_index=save_index, limit=limit, **kwargs)
