from datascience.data.synthetize.create_dataset import create_dataset
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import load_create_nn, FullyConnected


from datascience.visu.deep_test_plots import plot_decision_boundary_and_all
from datascience.visu.util.util import save_fig

# constructing the dataset
train, test = create_dataset(param_train=(250, 250), poly=True)

# creating/loading a model
model_params = {
    'architecture': (8,),  # play with config GD and SGD + architecture for the first figures
    'dropout': 0.0,
}
model = load_create_nn(model_class=FullyConnected, model_params=model_params)

# optimization
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

# plot results
plot_decision_boundary_and_all(train.dataset, train.labels, model)

save_fig()
