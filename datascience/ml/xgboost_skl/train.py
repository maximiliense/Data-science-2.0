import xgboost as xgb
import numpy as np
import ast

from datascience.ml.evaluation import validate, export_results
from engine.parameters import special_parameters
from engine.flags import incorrect_structure
from engine.logging import print_logs, print_h1, print_notif
from engine.core import module
from engine.path import output_path

# TODO : debbug and improve

@module
def fit(model, train, test, validation_only=False, export=False, training_params=None, export_params=None):
    training_params = {} if training_params is None else training_params
    export_params = {} if export_params is None else export_params

    dtest = xgb.DMatrix(np.asarray(test.get_vectors()), label=np.asarray(test.labels))
    X_test = np.asarray(test.get_vectors())
    y_test = np.asarray(test.labels)

    if not validation_only:
        print_h1('Training: ' + special_parameters.setup_name)
        print_logs("get vectors...")

        X_train = np.asarray(train.get_vectors())
        y_train = np.asarray(train.labels)

        eval_set = [(X_test, y_test)]

        print_logs("fit model...")

        model.fit(
            X_train,
            y_train,
            eval_metric='merror',
            early_stopping_rounds=20,
            eval_set=eval_set
        )

        print_logs("Save model...")
        complement = {'best_iteration': bst.best_ntree_limit}
        with open(output_path("model_complement.txt"), "w") as file:
            file.write(str(complement))
        bst.save_model(output_path("model"))
        bst.dump_model(output_path("model_dump"))

    else:
        print_logs("load model " + special_parameters.output_path("_model"))
        bst = xgb.Booster()
        bst.load_model(output_path("model"))
        with open(output_path("model_complement.txt"), "r") as file:
            st = file.read()
            complement = ast.literal_eval(st)
        if 'best_iteration' in complement:
            bst.best_ntree_limit = complement['best_iteration']

    print_h1('Validation/Export: ' + special_parameters.setup_name)
    predictions = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    res = validate(
        predictions, np.array(test.labels), training_params['metrics'] if 'metrics' in training_params else tuple(),
        final=True
    )
    print_notif(res, end='')
    if export:
        export_results(test, predictions, **export_params)
