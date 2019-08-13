from sklearn.model_selection import train_test_split

from datascience.data.loader.occurrence_loader import _occurrence_loader
from datascience.data.util.source_management import check_source
from engine.core import module
from engine.util.merge_dict import merge_smooth


@module
def occurrence_loader(dataset_class, source=None, validation_size=0.1, test_size=0.1,
                      label_name='Label', id_name='id', splitter=train_test_split, filters=tuple(),
                      online_filters=tuple(), postprocessing=tuple(), save_index='default', limit=None, **kwargs):
    if source is not None:
        r = check_source(source)
        merge_smooth(kwargs, r)

    return _occurrence_loader(dataset_class, validation_size=validation_size, test_size=test_size,
                              label_name=label_name, id_name=id_name, splitter=splitter, filters=filters,
                              online_filters=online_filters, postprocessing=postprocessing, save_index=save_index,
                              limit=limit, **kwargs)
