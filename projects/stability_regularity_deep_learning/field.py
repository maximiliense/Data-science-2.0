import numpy as np

from datascience.visu.util import plt, save_fig

# dataset
pos_x = np.random.normal(loc=0, scale=5, size=10)
pos_y = -2 * pos_x + 2
pos = np.stack((pos_x, pos_y)).transpose()
neg_x = np.random.normal(loc=0, scale=5, size=10)
neg_y = -2 * neg_x - 2
neg = np.stack((neg_x, neg_y)).transpose()


y = np.array([[1] if i < 10 else [0] for i in range(20)])

dataset = np.concatenate([pos, neg])

dataset = np.concatenate((dataset, y), axis=1)

beta_gd = np.array([[0.], [0.]])
beta_sgd = np.array([[0.], [0.]])

ax = plt('optim').gca()
ax.scatter(dataset[:, 0], dataset[:, 1])

save_fig()

# optimization


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(y_hat, y):
    eps = 0.001
    l = np.dot(np.log(y_hat.T+eps), y) + np.dot(np.log(1-y_hat.T+eps), 1-y)
    return -l


def gradient(y_hat, y, X):

    return ((y-y_hat) * X).sum(axis=0)/X.shape[0]


def gradient_descent(data, param, nb_iterations=500, batch_size=20, eta=0.1):

    i = 0
    y = data[:, 2:3]
    X = data[:, :2]

    hist = [np.copy(param)]
    deb = 0
    end = batch_size
    l = 0.
    while i < nb_iterations:
        i += 1
        X_to_use = X[deb:end, :2]
        y_to_use = y[deb:end]
        y_hat = sigmoid(np.dot(X_to_use, param))
        y_hat_complete = sigmoid(np.dot(X, param))
        # print(gradient(y_hat, y_to_use, X_to_use).reshape(param.shape), 'gradient')
        param = param + eta * gradient(y_hat, y_to_use, X_to_use).reshape(param.shape)
        l += loss(y_hat_complete, y)
        # print(i, loss(y_hat_complete, y))
        deb = (deb + batch_size) % X.shape[0]
        end = deb + batch_size
        hist.append(np.copy(param))

        if deb == 0:
            np.random.shuffle(dataset)
    print(l / nb_iterations, 'loss')
    return np.array(hist).reshape(len(hist), 2)


hist_1 = gradient_descent(dataset, param=beta_gd)
hist_2 = gradient_descent(dataset, param=beta_sgd, batch_size=1)

def plot():
    y = dataset[:, 2:3]
    X = dataset[:, :2]
    p = plt('energy_landscape')
    ax = p.gca()
    ax.set_xlim(-0.5, 8.)
    ax.set_ylim(-1, 3.)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    _X = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    zz = []
    for i in range(_X.shape[0]):
        y_hat = sigmoid(np.dot(X[:, :2], _X[i:i+1].T))

        zz.append(loss(y_hat, y))
    zz = np.array(zz).reshape(xx.shape)

    cp = ax.contourf(xx, yy, zz)
    ax.plot(hist_2[:, 0], hist_2[:, 1])
    ax.plot(hist_1[:, 0], hist_1[:, 1])
    plt('energy_landascape').gcf().colorbar(cp)
    # cset = ax.contour(zz, np.arange(-1, 1.5, 0.2), linewidths=2)
    save_fig()


plot()
