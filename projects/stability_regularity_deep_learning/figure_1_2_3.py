from datascience.data.synthetize.create_dataset import create_dataset
from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.loss.loss import HebbLoss
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import FullyConnectedDeepAnalysis
from datascience.ml.neural.checkpoints import create_model

from datascience.visu.deep_test_plots import plot_db_partitions_gradients, plot_separator, plot_db_partitions
from datascience.visu.util.util import save_fig, remove_axis

# constructing the dataset
train, test = create_dataset(param_train=(250, 250), poly=True)

# creating/loading a model
model_params = {
    'architecture': (10, 10, 10),  # play with config GD and SGD + architecture for the first figures
    'dropout': 0.0,
}
model = create_model(model_class=FullyConnectedDeepAnalysis, model_params=model_params)

# optimization
training_params = {
    'iterations': [120],
    'log_modulo': -1,
    'val_modulo': 1,
    # 'loss': HebbLoss()
}

optim_params = {
    'momentum': 0.0,
    'lr': 0.5,
}

validation_params = {
    'metrics': (ValidationAccuracy(1),)
}

fit(
    model, train=train, test=test, training_params=training_params, optim_params=optim_params,
    validation_params=validation_params
)

# plot results
ax = plot_db_partitions(train.dataset, train.labels, model, colorbar=False)  # plot_db_partitions_gradients

plot_separator(train.separator, ax=ax)

remove_axis()

save_fig()
