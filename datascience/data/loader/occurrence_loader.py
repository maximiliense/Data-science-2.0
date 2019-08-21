from sklearn.model_selection import train_test_split
import pandas as pd

from datascience.data.util.filters import filter_test, index_labels, online_filters_processing
from datascience.data.util.index import save_reversed_index, get_to_save, get_to_load, get_index, reverse_indexing
from datascience.data.model_selection.util import perform_split
from engine.parameters import special_parameters
from engine.path import output_path
from engine.logging import print_dataset_statistics


def labels_indexed_str(indexer):
    if type(indexer) is list:
        return str([len(k) for k in indexer])
    else:
        return str(len(indexer))


def get_label(r, label_name):
    return [int(r[1][label]) for label in label_name] if type(label_name) in (tuple, list) else int(r[1][label_name])


def index_init(save_index, label_name):
    if save_index in ('default', 'auto') and not special_parameters.from_scratch:
        if label_name is not None:
            save_index = 'load_and_save'
        else:
            save_index = 'load'
    return save_index


# TODO filters for RF for instance...
def _occurrence_loader(dataset_class, occurrences, validation_size=0.1, test_size=0.1, label_name='Label',
                       id_name='id', splitter=train_test_split, filters=tuple(), online_filters=tuple(),
                       postprocessing=tuple(), save_index='default', limit=None, source_name='unknown',
                       stop_filter=False, **kwargs):
    """
    returns a train and a test set
    :type stop_filter: object
    :param source_name:
    :param postprocessing: post processing functions to apply on datasets
    :param limit:
    :param save_index: True, 'save' or False or 'load_and_save'
    :param online_filters:
    :param filters:
    :param splitter:
    :param rasters:
    :param id_name:
    :param label_name:
    :param validation_size:
    :param occurrences:
    :param dataset_class:
    :param test_size:
    :return: train, val and test set, pytorch ready
    """
    # initialize index to a specific behaviour if save index is default
    save_index = index_init(save_index, label_name)

    labels_indexed_bis = None

    # load an existing index
    if get_to_load(save_index):
        path = output_path('index.json')
        labels_indexed_bis = reverse_indexing(get_index(path))  # loading index and reversing it

    # or create index if failed or did not have to load one
    if labels_indexed_bis is None:
        # the test is for multi-labels
        labels_indexed_bis = {} if type(label_name) is not tuple else [{} for _ in label_name]

    # do not load all the lines if their number is limited
    if limit is None:
        df = pd.read_csv(occurrences, header='infer', sep=';', low_memory=False)
    else:
        df = pd.read_csv(occurrences, header='infer', sep=';', low_memory=False, nrows=limit)

    # filters unwanted occurrences
    df = df[df.apply(lambda _row: not online_filters_processing(online_filters, _row), axis=1)]

    # set label to -1 if no label or index label
    if label_name is None:
        df['label'] = -1
    else:
        df['label'] = df[label_name].apply(lambda name: index_labels(labels_indexed_bis, name))

    ids = df[id_name].to_numpy()
    labels = df['label'].to_numpy()
    dataset = df[['Latitude', 'Longitude']].to_numpy()

    # if need to save index, save it
    if get_to_save(save_index):
        path = output_path('index.json')
        save_reversed_index(path, labels_indexed_bis)  # saving index after reversing it...

    columns = (labels, dataset, ids)
    # splitting train test
    train, test = perform_split(columns, test_size, splitter)

    # splitting validation
    train, val = perform_split(train, validation_size, splitter)

    # apply filters
    # for f in filters:  # TODO update filters taking into account the new structure
    #    f(*train, *val, *test)
    if test_size != 1 and label_name is not None and not stop_filter:
        # Filtering elements that are only in the test set
        test = filter_test((train[0], val[0]), *test)

    # train set
    train = dataset_class(*train, **kwargs)
    if hasattr(train, 'extractor'):
        ext = train.extractor
        kwargs['extractor'] = ext

    # test set
    test = dataset_class(*test, **kwargs)

    # validation set
    validation = dataset_class(*val, **kwargs)

    # apply special functions on datasets
    for process in postprocessing:
        process(train, validation, test)

    # print dataset statistics
    labels_size = labels_indexed_str(labels_indexed_bis) if label_name is not None else '0'
    print_dataset_statistics(
        len(train), len(validation), len(test), source_name, labels_size
    )

    return train, validation, test
