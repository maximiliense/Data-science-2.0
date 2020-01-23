from datascience.ml.metrics.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies, JustExportPredictions
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.sklearn.train import fit
from datascience.ml.sklearn.util import load_or_create
from datascience.model_selection import train_test_split_stratified
from engine.parameters.special_parameters import get_parameters
from sklearn.ensemble.forest import RandomForestClassifier

max_depth = get_parameters('max_depth', 12)
save = get_parameters('save', True)

# loading dataset
train, _, test = occurrence_loader(
    EnvironmentalDataset, source='gbif_taxref', validation_size=0, size_patch=1
)

model = load_or_create(RandomForestClassifier, n_estimators=100, max_depth=max_depth)

# training model
training_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies(),
                ValidationAccuracyRangeBySpecies(max_top_k=100, final_validation=True),
                ValidationAccuracyForAllSpecies(train=train, final_validation=True),
                JustExportPredictions(save_true_labels=True))
}
fit(model, train=train, test=test, training_params=training_params, save=save)
