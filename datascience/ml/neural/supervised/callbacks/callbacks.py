import torch
from numpy.linalg import norm
from torch.nn import Linear, BatchNorm1d, Conv2d
import numpy as np

from datascience.math import compute_filters
from datascience.ml.neural.models.cnn import CustomizableCNN
from datascience.ml.neural.models.fully_connected import FullyConnected
from datascience.ml.neural.supervised.callbacks.util import Callback
from datascience.visu.deep_test_plots import plot_dataset, plot_activation_rate, plot_decision_boundary, \
    plot_gradient_field, compute_neural_directions
from datascience.visu.util import plt, get_figure
from sklearn.decomposition import PCA

from datascience.visu.util.util import save_fig_direct_call
from engine.logging import print_warning, DataParallel


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
        plot_decision_boundary(self.model, ax=ax, figure_name='VCallback')
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


class FilterVarianceCallback(Callback):
    """
    Analysis the variance of the filter of 1 hidden layer convolutional network
    """
    def __init__(self, window_size=5, averaged=True, fig_name='FilterVarianceCallback'):
        super().__init__()
        self.window_size = window_size
        self.filters_history = []
        self.averaged = averaged
        self.fig_name = fig_name

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        assert (type(self.model) is CustomizableCNN), "VarianceCallback only works with CustomizableCNN."
        # assert (len(self.model.conv_layers) == 1), "VarianceCallback only works with 1 convolutional layer networks."

    def last_call(self):
        # transform data into matrix
        matrix = np.transpose(np.array(self.filters_history), (1, 0, 2))

        iterations = max(matrix.shape[1] - self.window_size + 1, 1)
        scatter_points = []
        # the number of iterations depend on the window size
        for i in range(iterations):
            current_time_window = matrix[:, i:(i+self.window_size), :]

            # for each window what is the average filter
            mean_filters = np.average(current_time_window, axis=1)
            variance = []
            # for each filter
            for j in range(mean_filters.shape[0]):
                one_filter = []
                variance.append(one_filter)

                # for each occurrence of the filter in the time window
                for z in range(current_time_window.shape[1]):
                    # dot product
                    one_filter.append(np.dot(current_time_window[j, z], mean_filters[j]))

            # compute the variance of the dot product per filter
            variance = np.var(np.array(variance), axis=1)
            if self.averaged:
                variance = [np.average(variance)]
            scatter_points.append(variance)
        scatter_points = np.array(scatter_points)
        print(scatter_points.shape)
        fig = get_figure(self.fig_name)
        ax = fig.gca()
        for i in range(scatter_points.shape[1]):
            ax.plot(scatter_points[:, i])
        save_fig_direct_call(figure_name=self.fig_name)

    def __call__(self, validation_id):
        self.filters_history.append(
            torch.flatten(self.model.conv_layers[0][0].weight, start_dim=1).detach().cpu().numpy()
        )


class ParameterVarianceCallback(Callback):
    """
    Analysis the variance of the filter of 1 hidden layer convolutional network
    """
    def __init__(self, window_size=5, averaged=True, fig_name='ParameterVarianceCallback'):
        super().__init__()
        self.window_size = window_size
        self.parameters_history = []
        self.averaged = averaged
        self.fig_name = fig_name

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        assert (type(self.model) is CustomizableCNN), "ParameterVarianceCallback only works with CustomizableCNN."
        # assert (len(self.model.conv_layers) == 1), "VarianceCallback only works with 1 convolutional layer networks."

    def last_call(self):
        # transform data into matrix

        matrix = np.array(self.parameters_history)
        matrix = matrix.reshape((matrix.shape[0], matrix.shape[1]))
        matrix = np.transpose(matrix, (1, 0))
        iterations = max(matrix.shape[1] - self.window_size + 1, 1)
        scatter_points = []
        # the number of iterations depend on the window size
        for i in range(iterations):
            variance = np.var(matrix[:, i:(i+self.window_size)], axis=1)

            if self.averaged:
                variance = [np.average(variance)]
            scatter_points.append(variance)
        scatter_points = np.array(scatter_points)

        fig = get_figure(self.fig_name)
        ax = fig.gca()
        for i in range(scatter_points.shape[1]):
            ax.plot(scatter_points[:, i])
        save_fig_direct_call(figure_name=self.fig_name)

    def __call__(self, validation_id):
        self.parameters_history.append(
            self.model.conv_layers[0][0].weight[:, 0, 0, 0].unsqueeze(dim=-1).detach().cpu().numpy()
        )


class NewStatCallback(Callback):
    def __init__(self, dataset=None):
        super().__init__()
        self.dir_variances = None
        self.shape = 100
        self.dataset = dataset

    def initial_call(self, modulo, nb_calls, dataset, model):
        """
        initialize the callback
        :param modulo:
        :param nb_calls:
        :param dataset:
        :param model:
        :return:
        """
        super().initial_call(modulo, nb_calls, dataset, model)
        self.dir_variances = []
        self.shape = self.dataset[0][0].shape[0] if len(self.dataset[0][0].shape) == 1 else 100
        model_instance = model.module if type(model) is DataParallel else model
        for _ in range(len(model_instance)):
            self.dir_variances.append([])

    def last_call(self):
        """
        Last call of the callback. The figure is generated here
        :return:
        """
        fig = get_figure('StatCallback')
        ax = fig.gca()
        for j, i in enumerate(self.dir_variances):
            ax.plot(i, label='Layer ' + str(j + 1))
        fig.legend()
        save_fig_direct_call(figure_name='StatCallback')

    def __call__(self, validation_id):
        directions = compute_filters(self.model, self.dataset)

        # we iterate over each layer
        for l, layer in enumerate(directions):
            vectors = layer - np.mean(layer, axis=0)
            pca_dir = PCA(n_components=self.shape)
            pca_dir.fit(vectors)

            var = -np.log(pca_dir.explained_variance_ratio_[0])
            self.dir_variances[l].append(var)
            print_warning("Layer %d: var = %f" % (l, var))
            print_warning("Norm 0: %d / %d = %f" % (np.mean(np.linalg.norm(layer,0,axis=1)),
                                                    vectors.shape[1],
                                                    np.mean(np.linalg.norm(layer, 0, axis=1))/vectors.shape[1]))


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

