from datascience.visu.util import save_fig
from datascience.visu.patch import pplot
from engine.parameters import get_parameters

patch_size = get_parameters('patch_size', 64)

pplot(latitude=45.667830, longitude=6.383380, source='glc18', size=patch_size, resolution=1, nb_cols=7, alpha=0.),

save_fig(extension='png')
