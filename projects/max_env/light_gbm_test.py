from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.light_gbm import fit
from datascience.ml.metrics import ValidationAccuracy

train, val, test = occurrence_loader(EnvironmentalDataset, source='glc18', limit=100, size_patch=1)

fit(train, test, val, validation_params={'metrics': (ValidationAccuracy(),)})
