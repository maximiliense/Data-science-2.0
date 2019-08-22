from engine.core import module
from engine.hardware import use_gpu, first_device
from engine.parameters.special_parameters import from_scratch, nb_workers
from engine.path.path import output_path
from engine.logging.verbosity import debug, verbose
from engine.logging.logs import print_debug, print_errors
from sklearn.externals import joblib
from xgboost import XGBClassifier


@module
def load_or_create(*args, **kwargs):
    if not use_gpu():
        print_errors('XGBoost can only be executed on a GPU for the moment', do_exit=True)
    if from_scratch:
        print_debug('Creating Random Forest Classifier')
        if debug:
            verbosity = 3
        elif verbose:
            verbosity = 2
        else:
            verbosity = 0
        gpu_id = first_device().index
        params = {'updater': 'grow_gpu', 'predictor': 'gpu_predictor', 'tree_method': 'gpu_hist',
                  'objective': 'multi:softprob', 'gpu_id': gpu_id}
        model = XGBClassifier(
            *args, **kwargs, verbosity=verbosity, n_jobs=nb_workers, seed=4242
        )
        model.set_params(**params)
    else:
        model = load_model()

    return model


def load_model(ext=''):
    path = output_path('models', ext + '_model.skl')
    print_debug('Loading SKL model ' + path)
    return joblib.load(path)


def save_model(model):
    path = output_path('models', '_model.skl')
    print_debug('Saving SKL model ' + path)
    return joblib.dump(model, path)
