import xgboost as xgb
import numpy as np
from datascience.ml.evaluation import validate, export_results
from datascience.ml.xgboost.util import save_model, load_model
from engine.flags import duplicated
from engine.parameters import special_parameters
from engine.logging import print_info, print_h1, print_notification, print_errors
from engine.core import module
from engine.logging.verbosity import verbose_level
from engine.hardware import use_gpu, first_device
from engine.parameters.special_parameters import validation_only


@module
@duplicated
def fit(train, test, export=False, training_params=None, export_params=None, **kwargs):
    if not use_gpu():
        print_errors('XGBoost can only be executed on a GPU for the moment', do_exit=True)

    training_params = {} if training_params is None else training_params
    export_params = {} if export_params is None else export_params

    d_test = xgb.DMatrix(np.asarray(test.get_vectors()), label=np.asarray(test.labels))

    if not validation_only:
        print_h1('Training: ' + special_parameters.setup_name)
        print_info("get vectors...")

        X = np.asarray(train.get_vectors())
        y = np.asarray(train.labels)

        d_train = xgb.DMatrix(X, label=y)

        gpu_id = first_device().index

        kwargs['verbosity'] = verbose_level()
        kwargs['gpu_id'] = gpu_id

        eval_list = [(d_test, 'eval'), (d_train, 'train')]

        print_info("fit model...")

        bst = xgb.train(
            kwargs,
            d_train,
            num_boost_round=kwargs["num_boost_round"],
            verbose_eval=kwargs["verbose_eval"],
            evals=eval_list
        )

        save_model(bst)

    else:
        bst = load_model()

    print_h1('Validation/Export: ' + special_parameters.setup_name)
    predictions = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)
    res = validate(
        predictions, np.array(test.labels), training_params['metrics'] if 'metrics' in training_params else tuple(),
        final=True
    )
    print_notification(res, end='')
    if export:
        export_results(test, predictions, **export_params)