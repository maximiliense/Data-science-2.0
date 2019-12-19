from datascience.ml.neural.loss import MSELoss
from engine.parameters import special_parameters
import torch.optim as optimizer

GAME_PARAMS = {

}

TRAINING_PARAMS = {
    'batch_size': 32,
    'iterations': [1000],
    'gamma': 0.1,
    'loss': MSELoss(),
    'val_modulo': 5,
    'log_modulo': 1,
    'first_epoch': special_parameters.first_epoch,
    'rm_size': 1000,
    'epsilon_start': 0.9,
    'epsilon_end': 0.1
}

PREDICT_PARAMS = {

}

VALIDATION_PARAMS = {
    'metrics': tuple()
}

EXPORT_PARAMS = {

}

OPTIM_PARAMS = {
    'momentum': 0.9,
    'weight_decay': 0,
    'optimizer': optimizer.SGD,
    'lr': 0.0005
}
