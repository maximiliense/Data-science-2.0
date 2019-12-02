from datascience.ml.metrics.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics.metrics import ValidationAccuracyMultiple, ValidationMRR
from datascience.ml.neural.models import InceptionEnv
from datascience.ml.neural.checkpoints import create_model
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from sklearn.model_selection import train_test_split
from projects.ecography.configs.inception import training_params, optim_params


model_params = {
    'dropout': 0.7,
    'n_labels': 1348,
    'n_input': 77
}

# loading/creating model
model = create_model(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, _, test = occurrence_loader(EnvironmentalDataset, source='glc19_pl_trusted', splitter=train_test_split, validation_size=0)

# training model
validation_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyMultiple([1, 10, 30]), ValidationMRR())
}

fit(
    model, train=train, test=test, training_params=training_params,
    validation_params=validation_params, optim_params=optim_params
)
