import math
import numpy as np
from pyproj import Proj, Transformer
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
from engine.path import output_path
from datascience.tools.ign.ign_image import IGNImageManager
import pandas as pd
from datascience.data.util.source_management import check_source
from engine.core import module
from engine.logging import print_info


@module
def create_ign_sparse(source_occ, source_ign, patch_size=64, error_path=output_path("error_extract/"),
                      **kwargs):

    r = check_source(source_occ)
    occurrences = r['occurrences']
    r = check_source(source_ign)
    ign_images = r['maps']

    la93 = Proj(init='epsg:2154')

    # extract manager
    im_manager = IGNImageManager(ign_images)

    # loading the occurrence file
    df = pd.read_csv(occurrences, header='infer', sep=';', low_memory=False)
    # df2 = df.copy()

    # sorting the dataset to optimise the extraction
    df.sort_values('Latitude', inplace=True)
    # df2.sort_values('Longitude', inplace=True)

    # p1 = df.head(100)
    # p2 = df.tail(100)
    # p3 = df2.head(100)
    # p4 = df2.tail(100)

    # new_df = pd.concat([p1, p2, p3, p4])
    # print(new_df)


    print_info(str(len(df)) + ' occurrences to extract!')

    im_manager.create_sparse(df[['Latitude', 'Longitude']], size=patch_size, step=1)

@module
def extract_from_ign_sparse(rasters=('red.npz', 'green.npz', 'blue.npz'), pos=(48.513870000000004, -4.764574), size=51,
                            res=5.0, top_left=(-360000, 7240000)):
    list_rasters = []
    for i in range(len(rasters)):
        list_rasters.append(load_npz(output_path(rasters[i])))

    modulo = size % 2
    half_size = int(size / 2)

    in_proj, ign_proj = Proj(init='epsg:4326'), Proj(init='epsg:2154')
    transformer = Transformer.from_proj(in_proj, ign_proj)

    y, x = transformer.transform(pos[1], pos[0])

    pos_x = int((top_left[1] - x) / res)
    pos_y = int((y - top_left[0]) / res)

    list_patch = []
    for i in range(len(list_rasters)):
        list_patch.append(list_rasters[i].tolil()[pos_x - half_size:pos_x + half_size + modulo,
        pos_y - half_size:pos_y + half_size + modulo].toarray())
        print(list_patch[i])
        print(type(list_patch[i]))
        print(list_patch[i].shape)

    patch = np.dstack(list_patch)
    print(patch)
    print(patch.shape)
    plt.imshow(patch)
    print_info("extraction test saved at: "+output_path("test_extract.png"))
    plt.savefig(output_path("test_extract.png"))


