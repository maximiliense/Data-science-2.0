from datascience.ml.metrics.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies,\
    ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.xgboost import fit
from engine.parameters.special_parameters import validation_only
from datascience.data.model_selection import train_test_split_stratified

# loading dataset
train, _, test = occurrence_loader(
    EnvironmentalDataset,
    source='gbif_taxref',
    id_name='id',
    label_name='Label',
    splitter=train_test_split_stratified,
    validation_size=0,
    size_patch=1
)

# training model
training_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyRangeBySpecies(max_top_k=100, final_validation=True),
                ValidationAccuracyForAllSpecies(train=train, final_validation=True))
}
fit(train=train, test=test, training_params=training_params, validation_only=validation_only)
