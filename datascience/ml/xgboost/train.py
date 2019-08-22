import xgboost as xgb
import numpy as np
from datascience.ml.evaluation import validate, export_results
from datascience.ml.xgboost.util import save_model, load_model
from engine.parameters import special_parameters
from engine.logging import print_logs, print_h1, print_notif
from engine.core import module


@module
def fit(train, test, validation_only=False, export=False, training_params=None, export_params=None):
    training_params = {} if training_params is None else training_params
    export_params = {} if export_params is None else export_params

    dtest = xgb.DMatrix(np.asarray(test.get_vectors()), label=np.asarray(test.labels))

    if not validation_only:
        print_h1('Training: ' + special_parameters.setup_name)
        print_logs("get vectors...")

        X = np.asarray(train.get_vectors())
        y = np.asarray(train.labels)

        dtrain = xgb.DMatrix(X, label=y)

        params = {'objective': 'multi:softprob', 'max_depth': 2, 'seed': 4242, 'silent': 0, 'eval_metric': 'merror',
                  'num_class': 6823, 'num_boost_round': 360, 'early_stopping_rounds': 10, 'verbose_eval': 1,
                  'updater': 'grow_gpu', 'predictor': 'gpu_predictor', 'tree_method': 'gpu_hist', 'nthread': 4}

        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        print_logs("fit model...")

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=params["num_boost_round"],
            verbose_eval=params["verbose_eval"],
            # feval=evaluator.evaluate,
            evals=evallist,
            # early_stopping_rounds=params["early_stopping_rounds"]
            # callbacks=[save_after_it]
        )

        print_logs("Save model...")
        save_model(bst)

    else:
        bst = load_model()

    print_h1('Validation/Export: ' + special_parameters.setup_name)
    predictions = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    res = validate(
        predictions, np.array(test.labels), training_params['metrics'] if 'metrics' in training_params else tuple(),
        final=True
    )
    print_notif(res, end='')
    if export:
        export_results(test, predictions, **export_params)