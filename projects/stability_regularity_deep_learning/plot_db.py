from datascience.data.synthetize import create_dataset
from datascience.ml.neural.models import FullyConnectedDeepAnalysis
from datascience.ml.neural.checkpoints import create_model
from datascience.visu.util import save_fig
from datascience.visu.deep_test_plots import plot_db_partitions_gradients

train, test = create_dataset(param_train=(100, 100), poly=True)

model_params = {
    'architecture': (5, 5, 5),
    'dropout': 0.0,
}

model = create_model(model_class=FullyConnectedDeepAnalysis, model_params=model_params)

plot_db_partitions_gradients(train.dataset, train.labels, model)

save_fig()
