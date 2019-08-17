from datascience.ml.neural.models import load_create_nn, InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalIGNDataset
from datascience.ml.neural.supervised import fit
from projects.max_env.configs.inception import training_params, validation_params, model_params


# creating environmental inception (more channels than classical inception)
model = load_create_nn(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, val, test = occurrence_loader(EnvironmentalIGNDataset, source='full_ign_5m')

# training model
fit(
    model, train=train, val=val, test=test, training_params=training_params, validation_params=validation_params
)
