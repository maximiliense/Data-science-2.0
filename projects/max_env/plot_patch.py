from datascience.visu.util import save_fig
from datascience.visu.patch import pplot

pplot(latitude=43.6, longitude=3.8, source='glc20', resolution=0.6, nb_cols=7, alpha=0.),

save_fig(extension='png')
