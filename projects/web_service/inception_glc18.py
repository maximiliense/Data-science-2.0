from datascience.ml.metrics.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics.metrics import ValidationAccuracyMultiple, ValidationMRR
from datascience.ml.neural.models import InceptionEnv
from datascience.ml.neural.checkpoints import create_model
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from sklearn.model_selection import train_test_split
from projects.ecography.configs.inception import model_params, training_params, optim_params

# loading/creating model
model = create_model(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, _, _ = occurrence_loader(EnvironmentalDataset, source='glc18', splitter=train_test_split, validation_size=0, test_size=0)
_, _, test = occurrence_loader(EnvironmentalDataset, source='glc18_test', splitter=train_test_split, validation_size=0, test_size=1)

# training model
validation_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyMultiple([1, 10, 30]), ValidationMRR())
}

fit(
    model, train=train, test=test, training_params=training_params,
    validation_params=validation_params, optim_params=optim_params
)
