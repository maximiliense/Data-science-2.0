from numpy.linalg import norm
from torch.nn import Linear, BatchNorm1d, Conv2d
import numpy as np

from datascience.math import compute_filters
from datascience.ml.neural.supervised.callbacks.util import Callback
from datascience.visu.deep_test_plots import plot_dataset, plot_activation_rate, plot_decision_boundary, \
    plot_gradient_field, compute_neural_directions
from datascience.visu.util import plt, get_figure
from sklearn.decomposition import PCA


class VCallback(Callback):

    def __init__(self, gradient_normalized=True):
        super().__init__()
        self.gradient_normalized = gradient_normalized
        self.fig = None

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        self.fig = plt('VCallback', figsize=(6.4, (nb_calls + 2) * 4.8))
        self.__call__(0)

    def last_call(self):
        self.__call__(self.nb_calls + 1)

    def __call__(self, validation_id):
        ax = plt('VCallback').subplot(self.nb_calls + 2, 1, validation_id + 1)
        plot_dataset(self.dataset.dataset, self.dataset.labels, ax=ax, figure_name='VCallback')
        plot_decision_boundary(self.dataset.dataset, self.dataset.labels, self.model, ax=ax,
                               figure_name='VCallback')
        plot_activation_rate(self.dataset.dataset, self.dataset.labels, self.model, ax=ax,
                             figure_name='VCallback')
        plot_gradient_field(self.dataset.dataset, self.dataset.labels, self.model, ax=ax, normalized=True,
                            figure_name='VCallback')


class StatCallback(Callback):
    def __init__(self):
        super().__init__()
        self.dir_variances = None

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        self.dir_variances = []
        for m in model.modules():
            if type(m) is Linear:
                self.dir_variances.append([])

    def last_call(self):
        fig = get_figure('StatCallback')
        ax = fig.gca()
        for j, i in enumerate(self.dir_variances):
            ax.plot(i, label='Layer ' + str(j + 1))
        fig.legend()

    def __call__(self, validation_id):
        directions, _, _ = compute_neural_directions(self.model, self.dataset.dataset, absolute_value=True,
                                                     threshold=False, min_activations=150)
        l = 0
        for m in self.model.modules():
            if type(m) is Linear and l < len(directions):
                vectors = np.vstack(directions[l])
                vectors = vectors - np.mean(vectors,axis=0)

                pca_dir = PCA(n_components=vectors.shape[1])
                pca_dir.fit(vectors)
                self.dir_variances[l].append(-np.log(pca_dir.explained_variance_ratio_[0]))

                l += 1


class NewStatCallback(Callback):
    def __init__(self):
        super().__init__()
        self.dir_variances = None

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        self.dir_variances = []
        for m in model.modules():
                self.dir_variances.append([])

    def last_call(self):
        fig = get_figure('StatCallback')
        ax = fig.gca()
        for j, i in enumerate(self.dir_variances):
            ax.plot(i, label='Layer ' + str(j + 1))
        fig.legend()

    def __call__(self, validation_id):
        directions = compute_filters( self.model, self.dataset)
        print("Callback !!!")
        # we iterate over each layer
        for l, layer in enumerate(directions):
            vectors = layer - np.mean(layer,axis=0)

            pca_dir = PCA(n_components=vectors.shape[1])
            pca_dir.fit(vectors)

            var = -np.log(pca_dir.explained_variance_ratio_[0])
            self.dir_variances[l].append(var)
            print("Layer %d: var = %f" % (l, var))



class GradientCallBack(Callback):
    def __init__(self):
        super().__init__()
        self.grad_mean = None
        self.grad_var = None

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        self.grad_mean = []
        weight_all_layer = []
        i=0
        for m in model.modules():
            if type(m) is Linear:
                self.grad_mean.append([])
                #weights = m.weight.detach().numpy()
                #self.grad_mean[i].append(np.mean(np.linalg.norm(weights, axis=0)))
                #weight_all_layer.append(np.hstack(weights))
                i += 1
        self.grad_mean.append([])
        #self.grad_mean[i].append(np.linalg.norm(np.concatenate(weight_all_layer)))

    def last_call(self):
        fig = get_figure('GradientCallBack')
        ax = fig.gca()
        for j, i in enumerate(self.grad_mean):
            ax.plot(i, label='Layer ' + str(j + 1))
        fig.legend()

    def __call__(self, validation_id):
        i = 0
        weight_all_layer = []
        for m in self.model.modules():
            if type(m) is Linear:
                weights = m.weight.detach().numpy()
                self.grad_mean[i].append(np.linalg.norm(np.hstack(weights)))
                weight_all_layer.append(np.hstack(weights))
                #grad = m.weight.grad.detach().numpy()
                #vectors = grad - np.mean(grad, axis=0)
                #pca_dir = PCA(n_components=vectors.shape[1])
                #pca_dir.fit(vectors)
                #self.grad_mean[i].append(np.linalg.norm(np.mean(grad, axis=0)))
                i += 1
        self.grad_mean[i].append(np.linalg.norm(np.concatenate(weight_all_layer)))

class Jacobian(Callback):
    def __init__(self):
        super().__init__()
        self.JJT_norm = None


    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        self.JJT_norm = []


    def last_call(self):
        fig = get_figure('JacobianCallBack')
        ax = fig.gca()
        ax.plot(self.JJT_norm)
        fig.legend()

    def __call__(self, validation_id):
        i = 0
        gradients = []
        for m in self.model.modules():
            if type(m) in (Linear, BatchNorm1d):
                gradients.append(m.weight.grad.detach().numpy())
                i += 1
        J = np.concatenate([r.flatten() for r in gradients])
        print(J.shape)
        JJT = np.outer(J,J)
        print(JJT.shape)
        self.JJT_norm.append(np.linalg.norm(JJT))

class ConvergenceCallBack(Callback):
    def __init__(self):
        super().__init__()
        self.last_state = None
        self.convergence = None

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        self.last_state = []
        self.convergence = []
        for m in model.modules():
            if type(m) is Linear:
                mat = m.weight.detach().numpy()
                self.last_state.append(np.copy(mat.reshape((mat.shape[0] * mat.shape[1], 1))))
                self.convergence.append([])

    def last_call(self):
        fig = get_figure('ConvergenceCallBack')
        ax = fig.gca()
        for j, i in enumerate(self.convergence):
            ax.plot(i, label='Layer ' + str(j + 1))
        fig.legend()

    def __call__(self, validation_id):
        i = 0
        for m in self.model.modules():
            if type(m) is Linear:
                mat = m.weight.detach().numpy()
                vect = mat.reshape((mat.shape[0]*mat.shape[1], 1))

                self.convergence[i].append(
                    1 - (np.dot(vect.T, self.last_state[i])/(norm(vect)*norm(self.last_state[i])))[0, 0]
                )

                self.last_state[i] = np.copy(vect)
                i += 1

