from datascience.ml.metrics.metrics import JustExportPredictions
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.sklearn.train import fit
from datascience.ml.sklearn.util import load_or_create
from datascience.model_selection import train_test_split_stratified
from sklearn.ensemble.forest import RandomForestClassifier
from engine.parameters.special_parameters import validation_only

if not validation_only:
    # loading dataset
    train, _, _ = occurrence_loader(
        EnvironmentalDataset, source='gbif_taxref', validation_size=0, size_patch=1, test_size=0
    )

else:
    train = None

_, _, test = occurrence_loader(
    EnvironmentalDataset, source='grid_occs_1km', validation_size=0, size_patch=1, test_size=1
)

model = load_or_create(RandomForestClassifier, n_estimators=100, max_depth=16)

# training model
training_params = {
    'metrics': (JustExportPredictions())
}

fit(model, train=train, test=test, training_params=training_params)