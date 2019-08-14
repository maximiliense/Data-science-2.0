from datascience.ml.metrics import ValidationAccuracy
from datascience.ml.neural.loss import CategoricalPoissonLoss
from projects.max_env.configs.inception import training_params, validation_params
from datascience.ml.neural.models import InceptionEnv, load_create_nn
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit

model_params = {
    'n_labels': 6823,
    'n_input': 77,
    'exp': True,  # poisson loss,
    'normalize_weight': 2.  # poisson loss
}

model = load_create_nn(model_class=InceptionEnv, model_params=model_params)


# loading dataset
train, val, test = occurrence_loader(
    EnvironmentalDataset,
    source='glc18',
    id_name='patch_id',
    label_name='species_glc_id',
    limit=1000
)

training_params['loss'] = CategoricalPoissonLoss()
training_params['log_modulo'] = 1
training_params['iterations'] = [10]
training_params['lr'] = 0.01

validation_params['metrics'] = (ValidationAccuracy(1),)  # let us just analyse convergence first

fit(model, train=train, val=val, test=test, training_params=training_params, validation_params=validation_params)
