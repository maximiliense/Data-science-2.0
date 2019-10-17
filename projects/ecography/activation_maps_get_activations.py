from datascience.ml.metrics import ValidationAccuracyMultipleBySpecies, ValidationMRRBySpecies
from datascience.ml.metrics import ValidationAccuracyRangeBySpecies, ValidationAccuracyForAllSpecies
from datascience.ml.neural.models import load_create_nn, InceptionEnv
from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.ml.neural.supervised import fit
from datascience.model_selection import train_test_split_stratified
from datascience.tools.activations_map.get_activations import get_species_neurons_activations

from projects.ecography.configs.inception import model_params

# loading/creating model
model = load_create_nn(model_class=InceptionEnv, model_params=model_params, from_scratch=False)

# loading dataset
_, _, grid_points = occurrence_loader(EnvironmentalDataset, source='grid_occs_1km', id_name='id',
                                      test_size=1, label_name=None)

# get activations
get_species_neurons_activations(model, grid_points)
