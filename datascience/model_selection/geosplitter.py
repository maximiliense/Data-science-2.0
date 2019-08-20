import numpy as np
from pyproj import Proj, Transformer
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time


class SplitterGeoQuadra(object):
    def __init__(self, quad_size=0, proj_in='epsg:4326', proj_out='epsg:3035'):
        self.quad_size = quad_size

        proj_in = Proj(init=proj_in)
        proj_out = Proj(init=proj_out)

        self.transformer = Transformer.from_proj(proj_in, proj_out)

    def _project(self, longitude, latitude):
        x, y = self.transformer.transform(longitude, latitude)
        return x / 1000, y / 1000

    def __call__(self, *columns, test_size, random_state=42):
        """
        perform the split. columns[1] must contains the GPS positions
        :param columns:
        :param test_size:
        :param random_state:
        :return:
        """
        # print(quad_size, columns)  if print is to stay, then use logging
        w = self.quad_size
        dataset = columns[1]
        ids = columns[2]

        if w == 0:
            train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
        else:
            r = np.random.RandomState(seed=random_state)
            d_lon = r.random_sample()
            r_lat = np.random.RandomState(seed=random_state + 10)
            d_lat = r_lat.random_sample()
            start = time.time()
            proj = OrderedDict()
            print('avant split')
            for i, coor in enumerate(dataset):
                lon, lat = self._project(coor[0], coor[1])
                proj_lon = (lon + d_lon * w) // w
                proj_lat = (lat + d_lat * w) // w
                if (proj_lon, proj_lat) in proj:
                    proj[(proj_lon, proj_lat)].append(ids[i])
                else:
                    proj[(proj_lon, proj_lat)] = [ids[i]]
            print(time.time() - start)
            start = time.time()
            train_map, test_map = train_test_split(list(proj.keys()), test_size=test_size, random_state=random_state)

            train_ids = [[i for i in proj[quadra]] for quadra in train_map]
            test_ids = [[i for i in proj[quadra]] for quadra in test_map]
            print(time.time() - start)
            print('apres les id')
        start = time.time()
        # creating output
        r = []
        for col in columns:
            r.extend((col[train_ids], col[test_ids]))
        print('apres les colonnes')
        print(time.time() - start)
        return r
