from engine.parameters.special_parameters import setup_name, validation_only
from engine.logging.logs import print_h1, print_info, print_notification
from datascience.ml.evaluation import validate, export_results
from datascience.ml.sklearn.util import save_model
from engine.core import module
import numpy as np


@module
def fit(model, train, test, export=False,
        training_params=None, export_params=None):
    training_params = {} if training_params is None else training_params
    export_params = {} if export_params is None else export_params
    clf = model
    if not validation_only:
        print_h1('Training: ' + setup_name)
        print_info("get vectors...")
        X = np.array(train.get_vectors())
        y = np.array(train.labels)

        print_info("fit model...")

        clf.fit(X, y)

        save_model(clf)
    print_h1('Validation/Export: ' + setup_name)
    restricted_predictions = clf.predict_proba(np.array(test.get_vectors()))
    nb_classes = max(clf.classes_)
    predictions = np.zeros((restricted_predictions.shape[0], nb_classes))
    predictions[:, clf.classes_] = restricted_predictions
    res = validate(
        predictions, np.array(test.labels), training_params['metrics'] if 'metrics' in training_params else tuple(),
        final=True
    )
    print_notification(res, end='')
    if export:
        export_results(test, predictions, **export_params)