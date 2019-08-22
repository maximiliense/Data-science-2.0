from engine.core import module
from engine.parameters.special_parameters import from_scratch, nb_workers
from engine.path.path import output_path
from engine.logging.verbosity import debug, verbose
from engine.logging.logs import print_debug
from sklearn.externals import joblib


@module
def load_or_create(model_class, *args, **kwargs):
    if from_scratch:
        print_debug('Creating Sklearn Model')
        verbosity = 2 if debug and verbose else 0
        model = model_class(
            *args, **kwargs, verbose=verbosity, n_jobs=nb_workers
        )
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
