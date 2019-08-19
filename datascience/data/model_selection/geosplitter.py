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


def splitter_geo_quadra(*columns, test_size, random_state=42, quad_size=0):
    print(quad_size, columns)
    w = quad_size
    dataset = columns[1]
    ids = columns[2]
    labels = columns[0]
    if w == 0:
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
            lon, lat = project(coor[0], coor[1])
            proj_lon = (lon + d_lon * w) // w
            proj_lat = (lat + d_lat * w) // w
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


def splitter_geo_quadra_diff_size(*columns, test_size, random_state=42, w):
        dataset = columns[0]
        ids = columns[1]
        labels = columns[2]
        x_te_samp = []
        ids_te_samp = []
        y_te_samp = []
        x_tr = dataset
        ids_tr = ids
        y_tr = labels

        for w_sample in w:
            x_tr, x_te, ids_tr, ids_te, y_tr, y_te = splitter_geo_quadra(x_tr, ids_tr, y_tr, w=w_sample,
                                                                test_size=test_size / len(w), random_state=random_state)
            x_te_samp = x_te_samp+x_te
            ids_te_samp = ids_te_samp + ids_te
            y_te_samp = y_te_samp + y_te
        res = x_tr, x_te_samp, ids_tr, ids_te_samp, y_tr, y_te_samp

        return res
