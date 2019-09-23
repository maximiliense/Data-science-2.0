from matplotlib.animation import FuncAnimation
import numpy as np

from datascience.ml.neural.supervised.callbacks.util import Callback
from datascience.visu.util.util import plt, get_figure, delete_figure
from engine.logging import print_info
from engine.path import output_path


class CircleCallback(Callback):

    def __init__(self, bias=False, log_norm=True, wk=True):
        super().__init__()
        self.parameters = []
        self.bias = []
        self.use_bias = bias
        self.log_norm = log_norm
        self.arrows = []
        self.wk = []
        self.use_wk = wk
        self.axis = None
        self.coef_norm = 1.

    def initial_call(self, modulo, nb_calls, dataset, model):
        super().initial_call(modulo, nb_calls, dataset, model)
        # self.__call__(0)

    def update(self, i):
        parameters = self.parameters[i]
        bias = self.bias[i]
        for j, p in enumerate(parameters):
            arrow = self.arrows[j]
            arrow.remove()
            p /= self.coef_norm
            norm = np.sqrt(p[0] ** 2 + p[1] ** 2)
            new_norm = norm * self.wk[i][j] if self.use_wk else norm
            b = -bias[j] if self.use_bias else 0.0
            b /= norm
            dx, dy = p[0] / norm * new_norm, p[1] / norm * new_norm

            x, y = (0, 0) if not self.use_bias else (p[0] * b / norm, p[1] * b / norm)

            self.arrows[j] = plt('circle').arrow(
                x, y, dx, dy, shape='full', head_width=0.04, head_length=0.08, fc='gray', ec='gray'
            )

    def last_call(self):
        step = 0.005
        x = np.arange(-1, 1. + step, step)
        y = np.sqrt(np.maximum(1. - x ** 2, np.zeros(x.shape)))
        plt('circle').plot(x, y)
        y = -np.sqrt(np.maximum(1. - x ** 2, np.zeros(x.shape)))
        plt('circle').plot(x, y)
        labels = self.dataset.labels
        dataset = self.dataset.dataset
        plt('circle').scatter(dataset[labels == 0][:, 0], dataset[labels == 0][:, 1])
        plt('circle').scatter(dataset[labels == 1][:, 0], dataset[labels == 1][:, 1])
        for i, p in enumerate(self.parameters[0]):
            norm = np.sqrt(p[0] ** 2 + p[1] ** 2)
            if norm > self.coef_norm:
                self.coef_norm = norm

        for i, p in enumerate(self.parameters[0]):
            p /= self.coef_norm
            norm = np.sqrt(p[0] ** 2 + p[1] ** 2)

            new_norm = norm * self.wk[0][i] if self.use_wk else norm

            b = -self.bias[0][i] if self.use_bias else 0.
            b /= norm
            dx, dy = p[0] * new_norm / norm, p[1] * new_norm / norm

            x, y = (0, 0) if not self.use_bias else (p[0] * b / norm, p[1] * b / norm)

            self.arrows.append(
                plt('circle').arrow(x, y, dx, dy, shape='full', head_width=0.04, head_length=0.08)
            )

        fig = get_figure('circle')
        self.axis = fig.gca()

        anim = FuncAnimation(fig, self.update, frames=np.arange(0, len(self.parameters)), interval=200)
        path = output_path('circle.gif')
        print_info('Saving GIF at ' + path)
        anim.save(path, dpi=80, writer='imagemagick')
        delete_figure('circle')

    def __call__(self, validation_id):
        model_params = list(self.model.parameters())
        self.parameters.append(np.copy(model_params[0].detach().cpu().numpy()))
        self.bias.append(np.copy(model_params[1].detach().cpu().numpy()))

        last_layers = np.copy(model_params[-2].detach().cpu().numpy())
        c = []

        for i in range(last_layers.shape[1]):
            # Je mets la norme l2 car c'est un softmax.. Pour prendre en compte les duex dimensions

            c.append(np.sqrt(last_layers[0][i]**2 + last_layers[1][i]**2))
        self.wk.append(c)
