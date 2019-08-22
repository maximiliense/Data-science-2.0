from datascience.ml.xgboost.util import load_model
from engine.core import module
from engine.parameters.special_parameters import from_scratch
from engine.logging.logs import print_debug
from xgboost import Booster


@module
def load_or_create(objective='multi:softprob', max_depth=2, seed=4242, eval_metric='merror',
                   num_class=4520, num_feature=256, **kwargs):

    if from_scratch:
        print_debug('Creating XGB Boosted Tree')
        params = {'updater': 'grow_gpu', 'predictor': 'gpu_predictor', 'tree_method': 'gpu_hist',
                  'eval_metric': eval_metric, 'objective': objective, 'num_class': num_class,
                  'max_depth': max_depth, 'seed': seed, 'num_feature': num_feature}

        params = {**params, **kwargs}

        model = Booster(
            params,
        )
    else:
        model = load_model()

    return model
