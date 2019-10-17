from engine.path.path import output_path
from engine.logging.logs import print_info
import xgboost as xgb
import ast


def load_model(model_name='model.xgb'):
    """
    Loading a model
    :return:
    """
    print_info("Loading model: " + output_path(model_name))
    bst = xgb.Booster()
    bst.load_model(output_path(model_name))
    with open(output_path("model_complement.txt"), "r") as file:
        st = file.read()
        complement = ast.literal_eval(st)
    if 'best_iteration' in complement:
        bst.best_ntree_limit = complement['best_iteration']
    return bst


def save_model(model, model_name='model.xgb'):
    print_info("Saving model: " + output_path(model_name))

    complement = {'best_iteration': model.best_ntree_limit}
    with open(output_path("model_complement.txt"), "w") as file:
        file.write(str(complement))

    model.save_model(output_path(model_name))
