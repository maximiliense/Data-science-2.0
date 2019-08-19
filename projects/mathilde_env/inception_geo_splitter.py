from projects.max_env.configs.inception import training_params, validation_params, model_params
from datascience.ml.neural.models import load_create_nn, InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.model_selection.geosplitter import splitter_geo_quadra
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit

# loading/creating model
model_params['n_input'] = 77
model_params['dropout'] = 0.75

model = load_create_nn(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, val, test = occurrence_loader(EnvironmentalDataset, source='glc18', splitter=splitter_geo_quadra, quad_size=10)

fit(model, train=train, val=val, test=test, training_params=training_params, validation_params=validation_params)
