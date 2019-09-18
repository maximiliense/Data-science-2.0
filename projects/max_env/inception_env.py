from datascience.ml.neural.checkpoints import create_model
from projects.max_env.configs.inception import training_params, validation_params, model_params, optim_params
from datascience.ml.neural.models import InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit

# loading/creating model
model_params['n_input'] = 77
model_params['dropout'] = 0.75

model = create_model(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, val, test = occurrence_loader(EnvironmentalDataset, source='glc19_fulldataset')

fit(model, train=train, val=val, test=test, training_params=training_params,
    validation_params=validation_params, optim_params=optim_params)
