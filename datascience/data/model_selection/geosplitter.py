import numpy as np
from pyproj import transform, Proj, Transformer
from itertools import chain
import random
from sklearn.model_selection import train_test_split
from collections import OrderedDict


class DatasetSplitter:
    pass


class DatasetSplitterGeoQuadra(DatasetSplitter):
    def __init__(self, w):
        self.w = w
        self.proj_in = Proj(init='epsg:4326')
        self.proj_out = Proj(init='epsg:3035')
        self.transformer = Transformer.from_proj(self.proj_in, self.proj_out)

    def project(self, longitude, latitude):
        x, y = self.transformer.transform(longitude, latitude)
        return x / 1000, y / 1000

    def __call__(self, *columns, test_size, random_state=42):
        dataset = columns[0]
        ids = columns[1]
        labels = columns[2]
        if self.w == 0:
            x_tr, x_te, ids_tr, ids_te, y_tr, y_te = train_test_split(dataset, ids, labels,
                                                                      test_size=test_size, random_state=random_state)
        else:
            r_lon = np.random.RandomState(random_state)
            d_lon = r_lon.rand()
            r_lat = np.random.RandomState(random_state + 10)
            d_lat = r_lat.rand()

            proj = OrderedDict()
            data = {}

            for i, coor in enumerate(dataset):
                lon, lat = self.project(coor[0], coor[1])
                proj_lon = (lon + d_lon * self.w) // self.w
                proj_lat = (lat + d_lat * self.w) // self.w
                if (proj_lon, proj_lat) in proj:
                    proj[(proj_lon, proj_lat)].append(ids[i])
                else:
                    proj[(proj_lon, proj_lat)] = [ids[i]]

                data[ids[i]] = (dataset[i], labels[i])

            train_map, test_map = train_test_split(list(proj.keys()), test_size=test_size, random_state=random_state)
            x_tr = []
            y_tr = []
            x_te = []
            y_te = []
            ids_tr = []
            ids_te = []

            for quadra in train_map:
                q_ids = proj[quadra]
                for id in q_ids:
                    x_tr.append(data[id][0])
                    y_tr.append(data[id][1])
                    ids_tr.append(id)

            for quadra in test_map:
                q_ids = proj[quadra]
                for id in q_ids:
                    x_te.append(data[id][0])
                    y_te.append(data[id][1])
                    ids_te.append(id)

        res = x_tr, x_te, ids_tr, ids_te, y_tr, y_te
        return res


class DatasetSplitterGeoQuadraDiffSize(DatasetSplitter):
    def __init__(self, w):
        self.w = w
        self.proj_in = Proj(init='epsg:4326')
        self.proj_out = Proj(init='epsg:3035')
        self.transformer = Transformer.from_proj(self.proj_in, self.proj_out)

    def project(self, longitude, latitude):
        x, y = self.transformer.transform(longitude, latitude)
        return x / 1000, y / 1000

    def split(self, dataset, ids, labels, w_sample, test_size_sample, random_state=42):
        r_lon = np.random.RandomState(random_state)
        d_lon = r_lon.rand()
        r_lat = np.random.RandomState(random_state + 10)
        d_lat = r_lat.rand()

        proj = OrderedDict()
        data = {}
        for i, coor in enumerate(dataset):
            lon, lat = self.project(coor[0], coor[1])
            proj_lon = (lon + d_lon * w_sample) // w_sample
            proj_lat = (lat + d_lat * w_sample) // w_sample
            if (proj_lon, proj_lat) in proj:
                proj[(proj_lon, proj_lat)].append(ids[i])
            else:
                proj[(proj_lon, proj_lat)] = [ids[i]]

            data[ids[i]] = (dataset[i], labels[i])

        train_map, test_map = train_test_split(list(proj.keys()), random_state=random_state, test_size=test_size_sample)
        x_tr = []
        y_tr = []
        x_te = []
        y_te = []
        ids_tr = []
        ids_te = []

        for quadra in train_map:
            q_ids = proj[quadra]
            for id in q_ids:
                x_tr.append(data[id][0])
                y_tr.append(data[id][1])
                ids_tr.append(id)

        for quadra in test_map:
            q_ids = proj[quadra]
            for id in q_ids:
                x_te.append(data[id][0])
                y_te.append(data[id][1])
                ids_te.append(id)

        return x_tr, x_te, ids_tr, ids_te, y_tr, y_te

    def __call__(self, *columns, test_size=0.1, random_state=42):
        dataset = columns[0]
        ids = columns[1]
        labels = columns[2]
        x_te_samp = []
        ids_te_samp = []
        y_te_samp = []
        x_tr = dataset
        ids_tr = ids
        y_tr = labels

        for w_sample in self.w:

            x_tr, x_te, ids_tr, ids_te, y_tr, y_te = self.split(x_tr, ids_tr, y_tr, w_sample=w_sample, test_size_sample=test_size / len(self.w),
                                                                    random_state=42)
            x_te_samp = x_te_samp+x_te
            ids_te_samp = ids_te_samp + ids_te
            y_te_samp = y_te_samp + y_te
        res = x_tr, x_te_samp, ids_tr, ids_te_samp, y_tr, y_te_samp

        return res
