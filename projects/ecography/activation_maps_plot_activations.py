from datascience.data.loader import occurrence_loader
from datascience.data.datasets import EnvironmentalDataset
from datascience.tools.activations_map.actiavtion_maps import plot_activations_on_map

# loading dataset
_, _, grid_points = occurrence_loader(EnvironmentalDataset, source='grid_occs_1km', id_name='id',
                                      test_size=1, label_name=None)

# get activations
plot_activations_on_map(grid_points)
