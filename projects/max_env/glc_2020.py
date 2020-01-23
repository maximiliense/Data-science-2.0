from datascience.data.rasters.environmental_raster_glc import PatchExtractor
from datascience.data.util.source_management import check_source
from datascience.visu.util import save_fig, plt

import os
import math
import numpy
r = check_source('glc20')
rasters = os.path.join(r['rasters'], 'alti', 'N43E004.hgt')

print(rasters)

#

siz = os.path.getsize(rasters)
dim = int(math.sqrt(siz/2))

assert dim*dim*2 == siz, 'Invalid file size'

data = numpy.fromfile(rasters, numpy.dtype('>i2'), dim*dim).reshape((dim, dim))

print(data.shape)

ax = plt('test').gca()
ax.imshow(numpy.log(data[:3500, :3500]))

save_fig()

exit()

extractor = PatchExtractor(rasters)
extractor.append('chbio_1')
extractor.plot(item=(43.6, 3.8), return_fig=True, nb_cols=1)
save_fig()
