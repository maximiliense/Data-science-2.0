from datascience.data.synthetize import create_dataset
from datascience.ml.neural.models import load_create_nn, FullyConnected
from datascience.visu.util import save_fig
from datascience.visu.deep_test_plots import plot_dataset, plot_activation_rate, plot_decision_boundary, \
    plot_gradient_field

train, test = create_dataset(param_train=(100, 100), poly=True)

model_params = {
    'architecture': (5, 5, 5),
    'dropout': 0.0,
    'batchnorm': True,
    'bias': True,
    'relu': True,
    'last_sigmoid': True
}
model = load_create_nn(model_class=FullyConnected, model_params=model_params)
print(model)

ax = plot_dataset(train.dataset, train.labels)
plot_activation_rate(train.dataset, train.labels, model, ax=ax)

plot_decision_boundary(train.dataset, train.labels, model, ax=ax)

plot_gradient_field(train.dataset, train.labels, model, ax=ax)

save_fig()
