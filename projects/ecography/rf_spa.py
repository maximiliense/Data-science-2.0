from datascience.ml.metrics.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.data.loader import occurrence_loader
from datascience.data.datasets.dataset_simple import GeoLifeClefDataset
from datascience.ml.sklearn.train import fit
from datascience.ml.sklearn.util import load_or_create
from datascience.model_selection import train_test_split_stratified
from sklearn.ensemble.forest import RandomForestClassifier

# loading dataset
train, _, test = occurrence_loader(
    GeoLifeClefDataset, source='gbif_taxref', validation_size=0, test_size=0
)

model = load_or_create(RandomForestClassifier, n_estimators=100, max_depth=17)

# training model
training_params = {
    'metrics': (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationMRRBySpecies())
}
fit(model, train=train, test=test, training_params=training_params)
