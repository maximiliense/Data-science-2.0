import torch

from datascience.data.synthetize.create_dataset import create_dataset
from datascience.ml.neural.supervised.callbacks.circle import CircleCallback
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import load_create_nn, FullyConnected

# creating dataset
from datascience.ml.neural.loss.loss import CELoss
from datascience.ml.metrics.metrics import ValidationAccuracy
import numpy as np


from datascience.visu.deep_test_plots import plot_decision_boundary, plot_dataset, plot_gradient_field, \
    plot_activation_rate
from datascience.visu.util.util import plt, save_fig
from engine.parameters import special_parameters

train, test = create_dataset(param_train=(250, 250), poly=True)
# creating/loading a model
model_params = {
    'architecture': (8,),  # play with config GD and SGD + architecture for the first figures
    'dropout': 0.0,
}

model = load_create_nn(model_class=FullyConnected, model_params=model_params)

training_params = {
    'lr': 0.1,
    'iterations': [50, 80, 100],
    'log_modulo': -1,
    'val_modulo': 1,
}

optim_params = {
    'momentum': 0.0
}

fit(
    model, train=train, test=test, training_params=training_params, optim_params=optim_params
)

ax = plot_dataset(train.dataset, train.labels)
plot_activation_rate(train.dataset, train.labels, model, ax=ax)

plot_decision_boundary(train.dataset, train.labels, model, ax=ax)

plot_gradient_field(train.dataset, train.labels, model, ax=ax)

save_fig()
