import numpy as np

from datascience.visu.util import plt, save_fig

arrow = np.array([np.random.normal(0, 1, size=(2,))])
mu_p = np.array([[0.], [1.]])
mu_m = np.array([[0.], [-1.]])
mu_k = mu_p if np.dot(mu_p.T, arrow[0]) > np.dot(mu_m.T, arrow[0]) else mu_m

eta = .01


def sigmoid(x):
    print(x)
    return 1 / (1 + np.exp(-x))


def alpha(t):
    print(t)
    return sigmoid(-0.0001*t+10)


alignment = []
norm = []

for i in range(1000):

    arrow[0] += eta * alpha(i) * mu_k.T[0]

    alignment.append(np.dot(mu_k.T, arrow[0])/(np.linalg.norm(mu_k)*np.linalg.norm(arrow[0])))
    norm.append(np.linalg.norm(arrow[0]))

ax = plt('alignment').gca()
ax.plot(alignment)
ax = plt('norm').gca()
ax.plot(norm)
save_fig()
