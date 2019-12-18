from datascience.ml.metrics.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.ml.neural.models import InceptionEnv
from datascience.ml.neural.checkpoints import create_model
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from sklearn.model_selection import train_test_split

from engine.parameters import get_parameters
from projects.web_service.configs.inception import training_params, optim_params


patch_size = get_parameters('patch_size', 64)
do = get_parameters('do', 0.7)

model_params = {
    'dropout': 0.5,
    'n_labels': 4520,
    'n_input': 77
}

# loading/creating model
model = create_model(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, val, test = occurrence_loader(EnvironmentalDataset, source='gbif_taxref',
                                     splitter=train_test_split, size_patch=patch_size)

# training model
validation_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyRangeBySpecies(max_top_k=100, final_validation=True),
                ValidationAccuracyForAllSpecies(train=train, final_validation=True))
}

fit(
    model, train=train, val=val, test=test, training_params=training_params,
    validation_params=validation_params, optim_params=optim_params
)
