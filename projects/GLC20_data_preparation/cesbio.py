import rasterio
import cv2
import os
import time as ti
import numpy as np
import scipy.misc
from pyproj import Transformer, Proj
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime
import pandas as pd

SIZE = 256
RESOLUTION = 1.0
CHECK_FILES = True


def _write_directory(root, *args):
    path = root
    for d in args:
        path = os.path.join(path, d)
        if not os.path.exists(path):
            os.makedirs(path)
    return path


def _print_details(idx, total, start, extract_time, latitude, longitude):
    time = datetime.datetime.now()
    print('\n{}/{}'.format(idx, total))
    p = ((idx - 1) / total) * 100
    print('%.2f' % p)
    delta = (time - start).total_seconds()
    estimation = (delta * total) / idx
    date_estimation = start + datetime.timedelta(seconds=estimation)
    print('mean extraction time: {}'.format(extract_time / idx))
    print('Actual position: {}'.format((latitude, longitude)))
    print('Time: {}, ETA: {}'.format(datetime.timedelta(seconds=delta), date_estimation))


class _ErrorManager(object):
    def __init__(self, path, cache_size=1000):
        # setting up the destination file
        current_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        error_extract_file = path + current_datetime + '_errors.csv'
        with open(error_extract_file, "w") as file:
            file.write(str(current_datetime) + "Occ_id;Latitude;Longitude\n")
        self.file = error_extract_file

        # setting up the error cache
        self.error_cache = []
        self.total_size = 0

        # the maximum number of elements in the cache
        self.cache_size = cache_size

    def __len__(self):
        return self.total_size

    def append(self, lat, lng, id):
        """
        :param error:
        :return:
        """

        self.error_cache.append('{};{};{}'.format(id, lat, lng))
        self.total_size += 1
        if len(self.error_cache) >= self.cache_size:
            self.write_errors()

    def write_errors(self):
        with open(self.file, "a") as file:
            for err in self.error_cache:
                file.write(err + "\n")
        self.error_cache = []


def extract_patch(lat, lon, transformer, res):
    pos_y, pos_x = transformer.transform(lon, lat)

    x = round((bounds.top - pos_x)/res)
    y = round((pos_y - bounds.left)/res)

    min_x = x - SIZE
    max_x = x + SIZE
    min_y = y - SIZE
    max_y = y + SIZE

    patch = arr[min_x:max_x, min_y:max_y]
    patch = np.repeat(np.repeat(patch, int(res), axis=1), int(res), axis=0)

    tick = int((2*res*SIZE)/(4*res))

    mean_tick = int(2*res)
    patch = patch[int((mean_tick-1)*tick):int((mean_tick+1)*tick), int((mean_tick-1)*tick):int((mean_tick+1)*tick)]
    return patch

with rasterio.open("/home/data/land_cover/cesbio/OCS_2016_CESBIO.tif") as t:
    bounds = t.bounds
    transfo = t.transform
    arr = t.read(1)

print("here")
print(arr.dtype)
arr[arr==11] = 1
arr[arr==12] = 2
arr[arr==31] = 3
arr[arr==32] = 4
arr[arr==34] = 5
arr[arr==36] = 6
arr[arr==41] = 7
arr[arr==42] = 8
arr[arr==43] = 9
arr[arr==44] = 10
arr[arr==45] = 11
arr[arr==46] = 12
arr[arr==51] = 13
arr[arr==53] = 14
arr[arr==211] = 15
arr[arr==221] = 16
arr[arr==222] = 17

# config geographic projections
in_proj, out_proj = Proj(init='epsg:4326'), Proj(init='epsg:2154')
transformer = Transformer.from_proj(in_proj, out_proj)

res = transfo[0]

# loading the occurrence file
df = pd.read_csv("/home/data/occurrences_GLC20/occurrences_GLC20_full_filtered.csv", header='infer', sep=';', low_memory=False)
df = df[['lon', 'lat', 'id']]

err_manager = _ErrorManager("/home/data/extract_GLC20/cesbio/")

if not os.path.exists('/home/data/extract_GLC20/cesbio/'):
    os.makedirs('/home/data/extract_GLC20/cesbio/')

total = df.shape[0]
start = datetime.datetime.now()
extract_time = 0

for idx, row in enumerate(df.iterrows()):
    longitude, latitude, occ_id = row[1][0], row[1][1], row[1][2]

    if idx % 10000 == 9999:
        _print_details(idx+1, total, start, extract_time, latitude, longitude)

    patch_id = int(row[1][2])

    # constructing path with hierarchical structure
    path = _write_directory('/home/data/extract_GLC20/cesbio/', str(patch_id)[-2:], str(patch_id)[-4:-2])

    patch_path = os.path.join(path, str(patch_id) + ".npy")

    # if file exists pursue extraction
    if os.path.isfile(patch_path) and CHECK_FILES:
        continue

    t1 = ti.time()
    t2 = 0
    try:
        patch = extract_patch(latitude, longitude, transformer, res)
    except Exception as err:
        t2 = ti.time()
        err_manager.append(latitude, longitude, occ_id)
    else:
        t2 = ti.time()
        np.save(patch_path, patch)
    finally:
        delta = t2 - t1
        extract_time += delta

    err_manager.write_errors()


print('extracted !')
