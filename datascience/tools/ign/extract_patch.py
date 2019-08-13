import datascience.tools.ign.ign_image as igni
import pandas as pd
import os
import numpy as np
import datetime
import time as ti
from pyproj import Transformer, Proj
from engine.core import module


@module
def extract_patch(offset=0, check_file=True):
    """
    Extract IGN patch from IGN maps.
    :param offset:
    :param check_file:
    :return:
    """
    im_manager = igni.IGNImageManager('/gpfswork/rech/fqg/uid61lx/data/ign_5m_maps/')  # on Jean Zay
    extract_size = 64
    extract_step = 1

    resolution = im_manager.image_resolution
    image_type = im_manager.image_type

    # d = "/data/ign/ign_patch_"+image_type+"_"+resolution+"_"+str(extract_size)+"_"+str(extract_step)+"/"

    d = '/gpfsssd/scratch/rech/fqg/uid61lx/data/ign_5m_patches/'

    if not os.path.exists(d):
        os.makedirs(d)

    df = pd.read_csv(
        '/gpfswork/rech/fqg/uid61lx/data/occurrences/full_ign.csv', header='infer', sep=';', low_memory=False
    )
    print(str(len(df)) + ' occurrences!')
    df.sort_values('Latitude', inplace=True)
    step_time = []

    total = df.shape[0]

    to_do = total-offset

    xm = im_manager.min_x
    xM = im_manager.max_x
    ym = im_manager.min_y
    yM = im_manager.max_y
    r = 1.2*im_manager.image_range

    extract_time = 0

    start = datetime.datetime.now()
    im_manager.init_error_file()
    print(start)
    proj_in = Proj(init='epsg:4326')
    proj_out = Proj(init='epsg:2154')
    transformer = Transformer.from_proj(proj_in, proj_out)
    # for idx, row in progressbar.progressbar(enumerate(df.iterrows())):
    for idx, row in enumerate(df.iterrows()):
        if idx >= offset:
            long, lat = row[1]['Longitude'], row[1]['Latitude']

            x2, y2 = transformer.transform(long, lat)

            if (idx - 1) % 100000 == 99999:
                time = datetime.datetime.now()
                print("\n"+str(idx)+"/"+str(total))
                p = ((idx - 1) / total) * 100
                print(str(p) + "%")
                delta = (time-start).total_seconds()
                estimation = (delta*to_do)/((idx - 1) - (offset + 1))
                date_estimation = start + datetime.timedelta(seconds=estimation)
                print("mean extraction time:", extract_time/idx)
                print("Actual position:", (lat, long), "Errors:", im_manager.nb_errors)
                print("Time:", datetime.timedelta(seconds=delta), "ETA:", date_estimation)

            if not(x2 < xm + r or x2 > xM - r or y2 < ym + r or y2 > yM - r):
                patch_id = int(row[1]['X_key'])
                # construcing path with hierachical structure
                sub_d = d + str(patch_id)[-2:] + "/"
                if not os.path.exists(sub_d):
                    os.makedirs(sub_d)
                sub_sub_d = sub_d + str(patch_id)[-4:-2] + "/"
                if not os.path.exists(sub_sub_d):
                    os.makedirs(sub_sub_d)

                # if file exists pursue extraction
                if os.path.isfile(sub_sub_d+str(patch_id)+".npy") and check_file:
                    continue

                t1 = ti.time()
                im = im_manager.extractPatch(x2, y2, extract_size, extract_step, id=int(patch_id))
                t2 = ti.time()
                delta = t2-t1
                extract_time += delta
                if im is not None:

                    np.save(sub_sub_d+str(patch_id)+".npy", im)
            else:
                pass

    im_manager.write_errors()
