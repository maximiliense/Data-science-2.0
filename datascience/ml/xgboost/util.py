from engine.core import module
from engine.hardware import use_gpu, first_device
from engine.parameters.special_parameters import from_scratch, nb_workers
from engine.path.path import output_path
from engine.logging.verbosity import debug, verbose
from engine.logging.logs import print_debug, print_errors, print_logs
from sklearn.externals import joblib
import xgboost as xgb
import ast


def load_model():
    print_logs("load model " + output_path("_model"))
    bst = xgb.Booster()
    bst.load_model(output_path("model"))
    with open(output_path("model_complement.txt"), "r") as file:
        st = file.read()
        complement = ast.literal_eval(st)
    if 'best_iteration' in complement:
        bst.best_ntree_limit = complement['best_iteration']
    return bst


def save_model(model):
    complement = {'best_iteration': model.best_ntree_limit}
    with open(output_path("model_complement.txt"), "w") as file:
        file.write(str(complement))
    model.save_model(output_path("model"))
    model.dump_model(output_path("model_dump"))
