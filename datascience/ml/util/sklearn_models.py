# RANDOM FOREST
from sklearn.externals import joblib

from engine.parameters import output_path_with_subdir
from engine.util.console import print_debug


def load_skl_model(ext=''):

    path = output_path_with_subdir('models', ext + '_model.skl')
    print_debug('Loading SKL model ' + path)
    return joblib.load(path)


def save_skl_model(model):
    path = output_path_with_subdir('models', '_model.skl')
    print_debug('Saving SKL model ' + path)
    return joblib.dump(model, path)
