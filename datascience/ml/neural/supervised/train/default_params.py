from datascience.ml.neural.loss import CELoss
from engine.parameters import special_parameters
import torch.optim as optimizer

TRAINING_PARAMS = {
    'batch_size': 32,
    'iterations': None,
    'gamma': 0.1,
    'loss': CELoss(),
    'val_modulo': 1,
    'log_modulo': -1,
    'first_epoch': special_parameters.first_epoch
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
    'lr': 0.1
}
