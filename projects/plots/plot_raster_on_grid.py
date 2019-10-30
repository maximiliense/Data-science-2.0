from datascience.data.loader import occurrence_loader
from datascience.data.datasets.environmental_dataset import EnvironmentalDataset
from datascience.visu.spatial_map_plots import plot_on_map
from engine.parameters.special_parameters import get_parameters
from datascience.data.rasters.environmental_raster_glc import raster_metadata
import numpy as np
import math

raster = get_parameters('raster', 'alti')


# loading dataset
_, _, grid_points = occurrence_loader(EnvironmentalDataset, source='grid_occs_1km', test_size=1,
                                         label_name=None, size_patch=1, add_all=False)

grid_points.extractor.append(raster)

r = np.zeros((len(grid_points.dataset), 1), dtype=float)
data_l = []

max = -2000
min = 10000
print(raster_metadata[raster]['nan'])

list_neg = []

for i, data in enumerate(grid_points.dataset):
    value = grid_points[i][0].numpy()
    if value[0] != raster_metadata[raster]['nan']:
        data_l.append([value[0]])
        #data_l.append([math.log(value[0]+50)])
    r[i, 0] = value[0]
    if value[0] > max:
        max_i = grid_points.ids[i]
        max = value[0]
    if value[0] < min:
        min_i = grid_points.ids[i]
        min = value[0]
    #r[i, 0] = math.log(value[0]+30)
    #r[i, 0] = min(value[0], 2000)

    if value[0] < 0:
        list_neg.append(value[0])
data_r = np.asarray(data_l)
print(min_i, min, max_i, max)
print(len(list_neg), i)
plot_on_map(data_r, grid_points.ids, n_cols=1, n_rows=1, figsize=20, log_scale=False,
            mean_size=1, selected=(0,), legend=(raster,), output=raster, bad_alpha=0, cmap='viridis')
