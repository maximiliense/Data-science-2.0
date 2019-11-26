from datascience.data.loader import occurrence_loader
from datascience.data.datasets.dataset_simple import GeoLifeClefDataset
from datascience.tools.activations_map.plot_activations_maps import plot_activations_on_map
from engine.parameters.special_parameters import get_parameters

n_rows = get_parameters('n_rows', 3)
n_cols = get_parameters('n_rows', 5)
mean_size = get_parameters('mean_size', 1)
figsize = get_parameters('figsize', 4)
log_scale = get_parameters('log_scale', False)
selected = get_parameters('selected', tuple())


# loading dataset
_, _, grid_points = occurrence_loader(GeoLifeClefDataset, source='grid_occs_1km', id_name='id',
                                      test_size=1, label_name=None)

# get activations
plot_activations_on_map(grid_points, n_rows=n_rows, n_cols=n_cols, log_scale=log_scale, figsize=figsize,
                        mean_size=mean_size, selected=selected)
