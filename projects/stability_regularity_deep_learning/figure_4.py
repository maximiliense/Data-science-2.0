from datascience.data.synthetize.create_dataset import create_dataset
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import FullyConnectedDeepAnalysis
from datascience.ml.neural.supervised.callbacks import NewStatCallback
from datascience.ml.neural.checkpoints import create_model

# constructing the dataset
train, test = create_dataset(param_train=(250, 250), poly=True)

# creating/loading a model
model_params = {
    'architecture': (10, 10, 10, 10),  # play with config GD and SGD + architecture for the first figures
    'dropout': 0.0,
}
model = create_model(model_class=FullyConnectedDeepAnalysis, model_params=model_params)

# optimization
training_params = {
    'lr': 0.1,
    'iterations': [50, 80, 100],
    'log_modulo': -1,
    'val_modulo': 1,
}

validation_params = {
    'vcallback': (NewStatCallback(),)
}

optim_params = {
    'momentum': 0.0,
    'lr': 0.1,
}

fit(
    model, train=train, test=test, training_params=training_params,
    optim_params=optim_params, validation_params=validation_params
)
