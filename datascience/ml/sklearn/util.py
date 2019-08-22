from engine.core import module
from engine.parameters.special_parameters import from_scratch, nb_workers
from engine.path.path import output_path
from engine.logging.verbosity import is_debug, is_verbose
from engine.logging.logs import print_debug
from sklearn.externals import joblib


@module
def load_or_create(model_class, model_name='model', *args, **kwargs):
    if from_scratch:
        print_debug('Creating SKL Model')

        model = model_class(
            *args, **kwargs, verbose=2 if is_debug() and is_verbose() else 0, n_jobs=nb_workers
        )

    else:
        model = load_model(model_name=model_name)

    return model


def load_model(model_name='model'):
    path = output_path('{}.skl'.format(model_name))
    print_debug('Loading SKL model: ' + path)
    return joblib.load(path)


def save_model(model, model_name='model'):
    path = output_path('{}.skl'.format(model_name))
    print_debug('Saving SKL model: ' + path)
    return joblib.dump(model, path)
