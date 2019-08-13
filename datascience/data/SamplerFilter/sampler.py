import math
import numpy as np
from sklearn.neighbors import BallTree
import time


def calcul_weight(input):
    w_i = {}
    tree = BallTree(np.radians(input.dataset), metric='haversine')
    ind = tree.query_radius(np.radians(input.dataset), r=2/6371, count_only=False)
    for idx, row in enumerate(input.ids):
        nb_oc = 0
        d_sp = 0
        for idx2, row2 in enumerate(ind[idx]):
            nb_oc += 1
            if input.labels[row2] == input.labels[idx]:
                d_sp += 1
        if nb_oc != 0 and d_sp != 0:
            w_i[row] = 1 / (nb_oc * d_sp)
        else:
            w_i[row] = 0
    return w_i
