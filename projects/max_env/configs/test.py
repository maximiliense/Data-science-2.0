
import os
from datascience.ml.evaluation.filters import FilterLabelsList

print(os.path.dirname(__file__) + '/allowed_classes_final.txt')

config = {
    'occurrence_loader': {
        'source': "glc19_test",
        'validation_size': 0.,
        'test_size': 1.
    },
    'fit': {
        'predict_params': {
                'filters': (FilterLabelsList(os.path.dirname(__file__) + '/allowed_classes_final.txt'),)
            }
    }
}
