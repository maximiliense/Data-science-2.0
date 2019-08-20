import numpy as np
from pyproj import Proj, Transformer
from sklearn.model_selection import train_test_split
from collections import OrderedDict


def project(longitude, latitude):
    proj_in = Proj(init='epsg:4326')
    proj_out = Proj(init='epsg:3035')
    transformer = Transformer.from_proj(proj_in, proj_out)
    x, y = transformer.transform(longitude, latitude)
    return x / 1000, y / 1000


class SplitterGeoQuadra(object):
    def __init__(self, quad_size=0):
        self.quad_size = quad_size

    def __call__(self, *columns, test_size, random_state=42):
        # print(quad_size, columns)  if print is to stay, then use logging
        w = self.quad_size
        dataset = columns[1]
        ids = columns[2]
        print(self.quad_size)
        if w == 0:
            train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
        else:
            r = np.random.RandomState(seed=random_state)
            d_lon = r.random_sample()
            r_lat = np.random.RandomState(seed=random_state + 10)
            d_lat = r_lat.random_sample()

            proj = {}
            print('avant split')
            for i, coor in enumerate(dataset):
                lon, lat = project(coor[0], coor[1])
                proj_lon = (lon + d_lon * w) // w
                proj_lat = (lat + d_lat * w) // w
                if (proj_lon, proj_lat) in proj.keys():
                    proj[(proj_lon, proj_lat)].append(ids[i])
                else:
                    proj[(proj_lon, proj_lat)] = [ids[i]]

            train_map, test_map = train_test_split(list(proj.keys()), test_size=test_size, random_state=random_state)

            train_ids = [[i for i in proj[quadra]] for quadra in train_map]
            test_ids = [[i for i in proj[quadra]] for quadra in test_map]
            print('apres les id')
        # creating output
        r = []
        for col in columns:
            r.extend((col[train_ids], col[test_ids]))
        print('apres les colonnes')
        return r
