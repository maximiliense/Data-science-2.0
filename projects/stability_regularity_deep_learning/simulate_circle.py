from datascience.data.synthetize import create_dataset
from datascience.visu.util import plt, save_fig

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

train, test = create_dataset(param_train=(250, 250, True, {'scale': 0.42}), poly=False)

nb_filters = 30.
eta = 0.5
B = 500
rho = 0.01


def pred_error(time):

    return 1.-np.exp(
        -eta/(2*B)*rho/(rho**2*time+(eta/(2*B)+0.1)**(1/rho)))


filters = np.array([
    [
        np.random.uniform(-1/np.sqrt(2), 1/np.sqrt(2)),
        np.random.uniform(-1/np.sqrt(2), 1/np.sqrt(2))
    ] for _ in range(int(nb_filters))])


def plot_circle(id_=1):
    ax = plt(str(id_)).gca()

    step = 0.005

    x = np.arange(-1, 1. + step, step)
    y = np.sqrt(np.maximum(1. - x ** 2, np.zeros(x.shape)))
    ax.plot(x, y)
    y = -np.sqrt(np.maximum(1. - x ** 2, np.zeros(x.shape)))
    ax.plot(x, y)
    def plot_filters():
        for f in filters:
            ax.arrow(0, 0, f[0]*0.2, f[1]*0.2, shape='full', head_width=0.04, head_length=0.08, fc='gray', ec='gray')

    plot_filters()


c = 0
print(filters.shape)
top = np.array([[0., 1.], [0., -1.]])
for t in range(100):
    print(pred_error(t))
    if t == 0 or t == 50 or t == 99:
        c += 1
        plot_circle(c)
    for f in filters:
        sim = cosine_similarity(f.reshape((1, 2)), top)
        if sim[0, 0] > sim[0, 1]:
            f += top[0] * 1/np.sqrt(nb_filters)*pred_error(t+1)
        else:
            f += top[1] * 1 / np.sqrt(nb_filters) * pred_error(t+1)

save_fig()
