import rasterio
import numpy as np
from pyproj import Transformer, Proj
import matplotlib.pyplot as plt
from matplotlib import colors

POSITION = (43.616195, 3.893479)
SIZE = 256
RESOLUTION = 1.0

with rasterio.open("/home/bdeneu/Downloads/OCS_2016_CESBIO.tif") as t:
    bounds = t.bounds
    transfo = t.transform
    arr = t.read(1)

# config geographic projections
in_proj, out_proj = Proj(init='epsg:4326'), Proj(init='epsg:2154')
transformer = Transformer.from_proj(in_proj, out_proj)

res = transfo[0]
pos_y, pos_x = transformer.transform(POSITION[1], POSITION[0])

x = round((bounds.top - pos_x)/res)
y = round((pos_y - bounds.left)/res)

min_x = x - 50
max_x = x + 50
min_y = y - 50
max_y = y + 50

patch = arr[min_x:max_x, min_y:max_y]


print(arr.shape)
print(np.max(arr))

cmap = colors.ListedColormap(['#ffa500', '#ffd700', '#316420', '#29514c', '#5eff00', '#b2d8d2', '0.25', '0.40', '0.60',
                              '0.75', '#cbbeb5', '#ffffb2', 'b', 'w', '#89cf1c', '#c00000', '#ff005e'])
parts = [0, 11.1, 12.1, 31.1, 32.1, 34.1, 36.1, 41.1, 42.1, 43.1, 44.1, 45.1, 46.1, 51.1, 53.1, 211.1, 221.1, 222.1]
norm = colors.BoundaryNorm(parts, cmap.N)
plt.imshow(patch, cmap=cmap, norm=norm)
plt.show()
