from datascience.ml.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.ml.neural.models import load_create_nn, InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from datascience.model_selection import train_test_split_stratified
from engine.parameters.special_parameters import get_parameters

from projects.ecography.configs.inception import model_params, training_params

temperature = get_parameters('temperature', 1.)

model_params['temperature'] = temperature

# loading/creating model
model = load_create_nn(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, val, test = occurrence_loader(EnvironmentalDataset, source='gbif_taxref', splitter=train_test_split_stratified)

# training model
validation_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyRangeBySpecies(max_top_k=100, final_validation=True),
                ValidationAccuracyForAllSpecies(train=train, final_validation=True))
}

fit(
    model, train=train, val=val, test=test, training_params=training_params, validation_params=validation_params
)
