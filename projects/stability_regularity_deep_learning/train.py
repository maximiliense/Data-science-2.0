import torch.optim as optimizer

from datascience.data.synthetize import create_dataset
from datascience.ml.metrics.metrics import ValidationAccuracy
from datascience.ml.neural.models import load_create_nn, FullyConnected
from datascience.ml.neural.supervised import fit
from datascience.visu.deep_test_plots import plot_dataset, plot_activation_rate, plot_decision_boundary, \
    plot_gradient_field
from datascience.visu.util import save_fig

train, test = create_dataset(param_train=(80, 80), poly=True)

model_params = {
    'architecture': (10, 10, 10),
    'dropout': 0.0,
    'batchnorm': True,
    'bias': True,
    'relu': True,
    'last_sigmoid': True
}
model = load_create_nn(model_class=FullyConnected, model_params=model_params)

training_params = {
    'lr': 0.1,
    'iterations': [100],
    'log_modulo': -1,
    'val_modulo': 1,
}
validation_params = {
    'vcallback': tuple(),
    'metrics': (ValidationAccuracy(1),)
}
optim_params = {
    'momentum': 0.0,
    'optimizer': optimizer.SGD
}
fit(
    model, train=train, test=test, training_params=training_params,
    validation_params=validation_params, optim_params=optim_params
)

ax = plot_dataset(train.dataset, train.labels)
plot_activation_rate(train.dataset, train.labels, model, ax=ax)

plot_decision_boundary(model, ax=ax)

plot_gradient_field(train.dataset, train.labels, model, ax=ax)

save_fig()

save_fig()
