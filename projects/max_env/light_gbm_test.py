from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.light_gbm import fit

train, val, test = occurrence_loader(EnvironmentalDataset, source='glc18', limit=100)

fit(train)
