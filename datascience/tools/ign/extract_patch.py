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

    # d = "/data/ign/ign_patch_"+image_type+"_"+resolution+"_"+str(extract_size)+"_"+str(extract_step)+"/"

    d = '/gpfsssd/scratch/rech/fqg/uid61lx/data/ign_5m_patches/'

    df = pd.read_csv(
        '/gpfswork/rech/fqg/uid61lx/data/occurrences/full_ign.csv', header='infer', sep=';', low_memory=False
    )

    print(str(len(df)) + ' occurrences!')
    df.sort_values('Latitude', inplace=True)

    list_pos = []
    list_ids = []

    for idx, row in enumerate(df.iterrows()):
        if idx >= offset:
            list_pos.append((row[1]['Longitude'], row[1]['Latitude']))
            list_ids.append(int(row[1]['X_key']))

    im_manager.extract_patches(list_pos, list_ids, d, size=extract_size, step=extract_step, check_file=check_file)
