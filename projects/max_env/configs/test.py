from sync.datascience.engine.parameters.special_parameters import get_setup_path

from datascience.ml.evaluation.filters import FilterLabelsList

config = {
    'occurrence_loader': {
        'source': "glc19_test",
        'validation_size': 0.,
        'test_size': 1.
    },
    'fit': {
        'predict_params': {
                'filters': (FilterLabelsList(get_setup_path() + '/allowed_classes_final.txt'),)
            }
    }
}
