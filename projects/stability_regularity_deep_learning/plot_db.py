from datascience.data.synthetize import create_dataset
from datascience.ml.neural.models import load_create_nn, FullyConnected
from datascience.visu.util import save_fig
from datascience.visu.deep_test_plots import plot_decision_boundary_and_all

train, test = create_dataset(param_train=(100, 100), poly=True)

model_params = {
    'architecture': (5, 5, 5),
    'dropout': 0.0,
}

model = load_create_nn(model_class=FullyConnected, model_params=model_params)

plot_decision_boundary_and_all(train.dataset, train.labels, model)

save_fig()
