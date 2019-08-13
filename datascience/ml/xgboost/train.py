import xgboost as xgb
import numpy as np
import ast

from datascience.ml.evaluation import validate, export_results
from engine.parameters import special_parameters
from engine.util.console.flags import incorrect_structure
from engine.util.console.logs import print_logs, print_h1, print_notif
from engine.core import module

# TODO separate load/save from fit... like for pytorch and sklearn
@module
@incorrect_structure(details='There should be a load and save functions, a load model module. This code should'
                             ' refer to the existing load function.')
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
                  'num_class': 6823, 'num_boost_round': 360, 'early_stopping_rounds': 10,
                  'verbose_eval': 1, 'updater': 'grow_gpu', 'predictor': 'gpu_predictor', 'tree_method': 'gpu_hist'}
        params['nthread'] = 4

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
        complement = {'best_iteration': bst.best_ntree_limit}
        with open(special_parameters.output_path("_model_complement.txt"), "w") as file:
            file.write(str(complement))
        bst.save_model(special_parameters.output_path("_model"))
        bst.dump_model(special_parameters.output_path("_model_dump"))

    else:
        print_logs("load model " + special_parameters.output_path("_model"))
        bst = xgb.Booster()
        bst.load_model(special_parameters.output_path("_model"))
        with open(special_parameters.output_path("_model_complement.txt"), "r") as file:
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
