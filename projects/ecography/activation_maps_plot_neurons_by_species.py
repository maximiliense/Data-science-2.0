from datascience.data.loader import occurrence_loader
from datascience.data.datasets.dataset_simple import GeoLifeClefDataset
from datascience.tools.activations_map.plot_activations_maps import select_neurons_by_species
from datascience.data.util.source_management import check_source
from engine.parameters.special_parameters import get_parameters

species = get_parameters('species', 0)
mean_size = get_parameters('mean_size', 1)
figsize = get_parameters('figsize', 5)

# loading dataset
_, _, grid_points = occurrence_loader(GeoLifeClefDataset, source='grid_occs_1km', id_name='id',
                                      test_size=1, label_name=None)

sources = check_source('gbif_taxref')


# get activations
select_neurons_by_species(grid_points, label_species=sources['label_species'], species=species, mean_size=mean_size,
                    figsize=figsize)