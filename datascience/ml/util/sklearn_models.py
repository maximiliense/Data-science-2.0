# RANDOM FOREST
from sklearn.externals import joblib

from engine.path import output_path
from engine.logging import print_debug


def load_skl_model(ext=''):

    path = output_path('models/' + ext + '_model.skl')
    print_debug('Loading SKL model ' + path)
    return joblib.load(path)


def save_skl_model(model):
    path = output_path('models/model.skl')
    print_debug('Saving SKL model ' + path)
    return joblib.dump(model, path)
