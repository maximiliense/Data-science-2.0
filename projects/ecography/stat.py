import numpy as np


def get_topk_species(predictions, labels, k=30):
    nb_labels = predictions.shape[1]
    result_sp = np.zeros(nb_labels)
    count = np.zeros(nb_labels)
    keep = np.zeros(nb_labels, dtype=bool)
    for i, pred in enumerate(predictions):
        rg = np.argwhere(pred == labels[i])[0, 0]
        count[labels[i]] += 1
        keep[labels[i]] = True
        if rg <= k:
            result_sp[labels[i]] += 1
    count = count[keep]
    result_sp = result_sp[np.array(keep)]
    return result_sp / count, result_sp.shape[0]

cnn = np.load("/home/benjamin/pycharm/Data-science-2.0/projects/ecography/inception_rs_normal_do07_4_20190831035247/predictions.npy")
cnn = np.argsort(-cnn, axis=1)
rf = np.load("/home/benjamin/pycharm/Data-science-2.0/projects/ecography/rf_env_final_d16_20190925145643/predictions.npy")
rf = np.argsort(-rf, axis=1)

#  bt = np.load("/home/benjamin/pycharm/Data-science-2.0/projects/ecography/bt_env_result_range_top100_by_species.npy")

print(cnn.shape)

test_labels = np.load("/home/benjamin/pycharm/Data-science-2.0/projects/ecography/inception_rs_normal_do07_4_20190831035247/true_labels.npy")

print(test_labels.shape)

list_z = []
for k in range(100):
    res, nb = get_topk_species(cnn, test_labels, k=k+1)
    m0 = get_topk_species(rf, test_labels, k=k+1)[0]
    # m0 = bt[k]

    sn = np.sqrt((1.0/(nb-1.0))*np.sum(np.square(res-np.mean(res))))

    z = np.sqrt(nb)*((np.mean(res) - np.mean(m0))/sn)
    list_z.append(z)

print(max(list_z))
print(min(list_z))

