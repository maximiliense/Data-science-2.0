import xgboost as xgb
import numpy as np
from datascience.ml.evaluation import validate, export_results
from datascience.ml.xgboost.util import save_model, load_model
from engine.parameters import special_parameters
from engine.logging import print_logs, print_h1, print_notif, print_errors
from engine.core import module
from engine.logging.verbosity import debug, verbose
from engine.hardware import use_gpu, first_device
from engine.parameters.special_parameters import validation_only
from engine.flags import deprecated, duplicated


@module
@deprecated(comment='The fonction is not working yet, please use the fit function in xgboost instead of xgboost_skl')
@duplicated
def fit(model, train, test, num_boost_round=360, verbose_eval=1, export=False, training_params=None, export_params=None, **kwargs):
    if not use_gpu():
        print_errors('XGBoost can only be executed on a GPU for the moment', do_exit=True)

    training_params = {} if training_params is None else training_params
    export_params = {} if export_params is None else export_params

    d_test = xgb.DMatrix(np.asarray(test.get_vectors()), label=np.asarray(test.labels))

    if not validation_only:
        print_h1('Training: ' + special_parameters.setup_name)
        print_logs("get vectors...")

        X = np.asarray(train.get_vectors())
        y = np.asarray(train.labels)

        d_train = xgb.DMatrix(X, label=y)

        if debug:
            verbosity = 3
        elif verbose:
            verbosity = 2
        else:
            verbosity = 0

        gpu_id = first_device().index

        kwargs['verbosity'] = verbosity
        kwargs['gpu_id'] = gpu_id

        eval_list = [(d_test, 'eval'), (d_train, 'train')]

        print_logs("fit model...")

        bst = xgb.train(
            kwargs,
            d_train,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
            evals=eval_list,
            xgb_model=model
        )

        print_logs("Save model...")
        save_model(bst)

    else:
        bst = load_model()

    print_h1('Validation/Export: ' + special_parameters.setup_name)
    predictions = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)
    res = validate(
        predictions, np.array(test.labels), training_params['metrics'] if 'metrics' in training_params else tuple(),
        final=True
    )
    print_notif(res, end='')
    if export:
        export_results(test, predictions, **export_params)