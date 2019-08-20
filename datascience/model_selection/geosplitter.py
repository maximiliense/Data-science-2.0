import numpy as np
from pyproj import Proj, Transformer
from sklearn.model_selection import train_test_split
from collections import OrderedDict


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
        labels = columns[0]
        print(self.quad_size)
        if w == 0:
            train_ids, test_ids = train_test_split(labels, test_size=test_size, random_state=random_state)
        else:
            r = np.random.RandomState(random_state)
            d_lon = r.random_sample()
            r_lat = np.random.RandomState(random_state + 10)
            d_lat = r_lat.random_sample()

            proj = OrderedDict()

            for i, coor in enumerate(dataset):
                lon, lat = self.project(coor[0], coor[1])
                proj_lon = (lon + d_lon * w) // w
                proj_lat = (lat + d_lat * w) // w
                if (proj_lon, proj_lat) in proj:
                    proj[(proj_lon, proj_lat)].append(labels[i])
                else:
                    proj[(proj_lon, proj_lat)] = [labels[i]]

            train_map, test_map = train_test_split(list(proj.keys()), test_size=test_size, random_state=random_state)

            train_ids = [i for quadra in train_map for i in proj[quadra]]
            test_ids = [i for quadra in test_map for i in proj[quadra]]
        # creating output
        r = []
        for col in columns:
            r.extend((col[train_ids], col[test_ids]))

        return r
