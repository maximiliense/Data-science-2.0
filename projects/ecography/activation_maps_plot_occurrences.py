from datascience.data.loader import occurrence_loader
from datascience.data.datasets.dataset_simple import GeoLifeClefDataset
from datascience.tools.activations_map.plot_activations_maps import species_train_test_occurrences
from datascience.data.util.source_management import check_source
from engine.parameters.special_parameters import get_parameters

species = get_parameters('species', 0)

# loading dataset
train, val, test = occurrence_loader(
        GeoLifeClefDataset, source='gbif_taxref', validation_size=0.1, size_patch=1, test_size=0.1
    )

sources = check_source('gbif_taxref')

species_train_test_occurrences(sources['label_species'], train, val, test, species=species)