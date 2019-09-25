from datascience.ml.neural.models import InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalIGNDataset
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.checkpoints import create_model
from projects.max_env.configs.inception import training_params, validation_params, model_params, optim_params

# creating environmental inception (more channels than classical inception)
model = create_model(model_class=InceptionEnv, model_params=model_params)

# loading dataset
train, val, test = occurrence_loader(EnvironmentalIGNDataset, source='full_ign_5m')

# memory issue on full_ign_5m due to size
test.limit = 2000

# training model
fit(
    model, train=train, val=val, test=test, training_params=training_params,
    validation_params=validation_params, optim_params=optim_params
)
