from datascience.tools.activations_map.plot_activations_maps import get_correlation_csv
from datascience.data.util.source_management import check_source

sources = check_source('gbif_taxref')


# get activations
get_correlation_csv(label_species=sources['label_species'])
