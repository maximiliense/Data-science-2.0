import numpy as np
import torch
from torch.nn import Linear, BatchNorm1d

from datascience.ml.neural.models import fully_connected
from datascience.visu.util.util import plt
from engine.util.console.logs import print_info, print_errors, print_logs
from engine.core import module


def plot(dataset, label, separator, clf=None, x_min=-0.5, x_max=0.55, y_min=-1, y_max=1.6,
         step_x=0.00125, step_y=0.00625):
    plt('plot_', figsize=(16, 14))
    if clf is not None:
        # mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))
        data = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

        Z = clf(data)
        Z = Z.argmax(dim=1)
        # Z = (Z > 0.5).int()
        Z = Z.reshape(xx.shape)
        ax = plt.gca()
        out = ax.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.3)
    plt('plot_').plot(separator[0, :], separator[1, :])
    plt('plot_').scatter(dataset[:, 0], dataset[:, 1], cmap=plt.cm.coolwarm, c=np.array(label))


@module
def plot_decision_boundary(X, y, model, figure_name='pdb', ax=None):
    pred_func = make_pred_func(model)
    if ax is None:
        ax = plt(figure_name).gca()
    if not callable(pred_func) and hasattr(pred_func, 'predict'):
        pred_func = pred_func.predict

    # plot_dataset(X, y, ax=ax)

    num = 1000
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
                         np.linspace(y_min, y_max, num))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=.4, zorder=-1)
    return ax


@module
def plot_dataset_module(X, y, ax=None, figure_name='plot_dataset', **kwargs):
    return plot_dataset(X, y, ax, figure_name, **kwargs)


def plot_dataset(X, y, ax=None, figure_name='plot_dataset', **kwargs):
    if ax is None:
        ax = plt(figure_name).gca()

    n_classes = len(np.unique(y))
    colors = ['red', 'cyan', 'orange']
    for k in range(n_classes):
        X_k = X[y == k]
        ax.scatter(X_k[:, 0], X_k[:, 1], color=colors[k], alpha=0.5)
    return ax


def plot_partition_distribution(model, X, figure_name='plot_pat_dist', ax=None):
    if ax is None:
        fig = plt(figure_name).figure()
        ax = fig.gca()

    _, activations = get_activations(model, X)
    unique, counts = np.unique(activations, return_counts=True, axis=0)
    print(len(unique))

    sorted_counts = np.sort(counts)[::-1]
    cumsum = sorted_counts.cumsum() / sorted_counts.sum()

    ax.plot(cumsum)
    ax.grid()


def make_dec_func(model):
    def dec_func(X):
        data = torch.from_numpy(X).float()
        Z = model(data)
        return Z

    return dec_func


def make_pred_func(model):
    dec_func = make_dec_func(model)

    def pred_func(X):
        output = dec_func(X)
        return output.argmax(dim=-1)

    return pred_func


def get_activations(model, X, layer=None):
    hook = ActivatedNeuronsForwardPreHook(layer=layer)
    hook.register(model)

    data = torch.from_numpy(X).float()
    _ = model(data)

    activations = hook.finalize()
    activations = np.hstack(activations)
    hook.remove()

    _, ind = np.unique(activations, return_inverse=True, axis=0)

    return activations, ind


class ActivatedNeuronsForwardPreHook:
    def __init__(self, layer=None):
        self.activated_neurons = {}
        self.layer = layer
        self.handles = None

    def __call__(self, m, input):
        res = (input[0].detach().cpu().numpy().copy() >= 0).astype(np.int)
        if m not in self.activated_neurons:
            self.activated_neurons[m] = []
        self.activated_neurons[m].append(res)

    def register(self, model, func=torch.nn.ReLU):
        self.handles = {}
        relu_count = 0
        for m in model.modules():
            if isinstance(m, func):
                relu_count += 1
                if self.layer is None or self.layer == relu_count:
                    handle = m.register_forward_pre_hook(self)
                    self.handles[m] = handle

    def remove(self):
        for module in list(self.handles):
            self.handles.pop(module).remove()

    def finalize(self):
        results = []
        for m in self.activated_neurons:
            results.append(np.vstack(self.activated_neurons[m]))
        return results


@module
def plot_gradient_field(X, y, model, ax=None, figure_name='pfg', normalized=True):
    from torch.autograd import Variable
    if ax is None:
        ax = plt(figure_name).gca()

    # disabling grad
    grads = []
    model.eval()
    for param in model.parameters():
        grads.append(param.requires_grad)
        param.requires_grad = False

    num = 20
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
                         np.linspace(y_min, y_max, num))

    _X = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    data = torch.from_numpy(_X).float()

    d = Variable(data, requires_grad=True)
    output = model(d)

    for i in range(d.shape[0]):
        output[i][0].backward(retain_graph=True)

    # plotting
    U = d.grad[:, 0]
    V = d.grad[:, 1]
    U = U.reshape(xx.shape)
    V = V.reshape(yy.shape)
    E_norm = 2 * np.sqrt(U ** 2 + V ** 2)
    if normalized:
        U = U/E_norm
        V = V / E_norm
        ax.quiver(xx, yy, U, V, E_norm, cmap=plt().cm.coolwarm, units='xy', scale=5.)
    else:
        ax.quiver(xx, yy, U, V, color='darkred', units='xy', scale=5.)
    # reestablishing grad
    for grad, param in zip(grads, model.parameters()):
        param.requires_grad = grad
    return ax


from PIL import Image, ImageFilter


@module
def plot_partition(X, y, model, figure_name='pp', ax=None):
    if ax is None:
        ax = plt(figure_name).gca()

    plot_dataset(X, y, ax=ax, alpha=.25)

    num = 500
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
                         np.linspace(y_min, y_max, num))
    _X = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    _, Z = get_activations(model, _X)
    Z = Z.reshape(xx.shape)

    levels = len(np.unique(Z))

    im = Image.fromarray(((Z / levels) * 255.).astype(np.uint8), mode='L')
    im = im.filter(ImageFilter.FIND_EDGES)
    im = im.point(lambda p: p > 0 and 255)
    ax.imshow(im, origin='lower', extent=(x_min, x_max, y_min, y_max), cmap='gray', aspect='auto')

    ax.imshow(Z / levels, origin='lower', extent=(x_min, x_max, y_min, y_max), alpha=.75, cmap='gray', aspect='auto')
    return ax


@module
def plot_activation_rate(X, y, model, ax=None, layer=None, figure_name='par', colorbar=True):
    if ax is None:
        ax = plt(figure_name).gca()

    # plot_dataset(X, y, ax=ax, alpha=.25)

    num = 500
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
                         np.linspace(y_min, y_max, num))
    _X = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    Z, _ = get_activations(model, _X, layer=layer)
    Z = np.average(Z, axis=1)

    Z = Z.reshape(xx.shape)
    c = ax.contourf(xx, yy, Z, cmap='gray', vmin=Z.min(), vmax=Z.max(), alpha=.5)
    if colorbar:
        ax.figure.colorbar(c, ax=ax, alpha=.75)
    return ax


def plot_decision_functions(X, y, dec_func, ax):
    if not callable(dec_func) and hasattr(dec_func, 'decision_function'):
        dec_func = dec_func.decision_function

    plot_dataset(X, y, ax=ax)

    num = 1000
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
                         np.linspace(y_min, y_max, num))
    Z = dec_func(np.c_[xx.ravel(), yy.ravel()]).detach().numpy()
    zz = Z[:, 0].reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.8, zorder=-1)


def plot_layer_output_corr(model, X, absolute_value, threshold, figure_name='ploc'):
    # this method only works on fully connected models
    if type(model) is not fully_connected.Net:
        print_errors(str(type(model)) + ' must be of type ' + str(fully_connected.Net) + '.', do_exit=True)

    layers = [m for m in model.modules() if type(m) in (BatchNorm1d, Linear)][:-1]
    final_layers = []
    it = 0

    while it < len(layers):
        # linear layer
        M = layers[it]
        it += 1

        linear_app = M.weight.detach().numpy()

        if it < len(layers) and type(layers[it]) is BatchNorm1d:
            A = layers[it]
            var = np.diag(A.running_var.numpy())

            gamma = np.diag(A.weight.detach().numpy())
            bn = np.matmul(gamma, np.linalg.inv(var))

            linear_app = np.matmul(bn, linear_app)
            it += 1
        final_layers.append(linear_app)

    # get activations
    activations, _ = get_activations(model, X)
    activations, _ = np.unique(activations, return_inverse=True, axis=0)

    # partition that are not on the domain are removed.

    for i, v in enumerate(np.all(activations == activations[0, :], axis=0)):
        if v:
            activations[:, i] = 0
    # unique after corrections
    activations, _ = np.unique(activations, return_inverse=True, axis=0)

    min_activations = 10

    nb_c_activations = min(activations.shape[0], min_activations)

    plt(figure_name, figsize=(nb_c_activations * 6.4, len(final_layers) * 4.8))
    vmin = 0. if absolute_value else -1.
    vmax = 1.
    for i in range(min_activations):
        la = None
        for li, l in enumerate(final_layers):
            cos = np.zeros((l.shape[0], l.shape[0]))
            activated = activations[i][li * l.shape[0]: (li+1) * l.shape[0]]

            if la is None:
                la = final_layers[li] * activated[:, np.newaxis]
            else:
                la = np.matmul(final_layers[li], la) * activated[:, np.newaxis]
            for r in range(l.shape[0]):
                for c in range(l.shape[0]):
                    if activations[i, li * l.shape[0] + r] != 0 and activations[i, li * l.shape[0] + c] != 0:
                        cos[r, c] = np.dot(la[r, :], la[c, :]) / np.linalg.norm(la[r, :]) / np.linalg.norm(la[c, :])
                    else:
                        cos[r, c] = None
            if absolute_value:
                cos = np.abs(cos)
            if type(threshold) is not bool:
                cos = (cos > threshold).astype(int)
            plt(figure_name).subplot(len(final_layers), nb_c_activations, 1 + li * nb_c_activations + i)
            plt(figure_name).imshow(cos, vmin=vmin, vmax=vmax)
            plt(figure_name).title('Layer: '+str(li + 1))
            plt(figure_name).colorbar()


def plot_corr(model, X, absolute_value, threshold, figure_name='pcr'):
    activations, _ = get_activations(model, X)
    activations, _ = np.unique(activations, return_inverse=True, axis=0)

    min_activations = 10

    # partition that are not on the domain are removed.

    for i, v in enumerate(np.all(activations == activations[0, :], axis=0)):
        if v:
            activations[:, i] = 0
    # unique after corrections
    activations, _ = np.unique(activations, return_inverse=True, axis=0)

    nb_activations = activations.shape[0]
    nb_c_activations = min(nb_activations, min_activations)

    nb_params = -1
    for n, p in model.named_parameters():
        if len(p.shape) == 2:
            nb_params += 1
    plt(figure_name, figsize=(nb_c_activations * 6.4, nb_params * 4.8))

    vmin = 0. if absolute_value else -1.
    vmax = 1.
    print_info(str(nb_activations) + ' affine spaces used')
    for idx, a in enumerate(range(nb_c_activations)):
        tc = 0
        c = 0
        for name, params in model.named_parameters():

            if len(params.shape) == 2:

                A = params.detach().numpy()
                B = np.zeros((A.shape[0], A.shape[0]))
                plt(figure_name).subplot(nb_params, nb_c_activations, 1 + c * nb_c_activations + idx)
                for i in range(A.shape[0]):

                    for j in range(A.shape[0]):
                        if activations[idx, tc + i] != 0 and activations[idx, tc + j] != 0:
                            B[i, j] = np.dot(A[i, :], A[j, :]) / np.linalg.norm(A[i, :]) / np.linalg.norm(A[j, :])
                        else:
                            B[i, j] = None
                if absolute_value:
                    B = np.abs(B)
                if type(threshold) is not bool:
                    B = (B > threshold).astype(int)
                plt(figure_name).imshow(B, vmin=vmin, vmax=vmax)
                plt(figure_name).title(name)
                plt(figure_name).colorbar()
                tc += A.shape[0]
                c += 1
            if tc >= activations.shape[1]:
                break


def compute_neural_directions(model, X, absolute_value, threshold, min_activations=10):
    # this method only works on fully connected models
    if type(model) is not fully_connected.Net:
        print_errors(str(type(model)) + ' must be of type ' + str(fully_connected.Net) + '.', do_exit=True)

    layers = [m for m in model.modules() if type(m) in (BatchNorm1d, Linear)][:-1]
    final_layers = []
    it = 0

    while it < len(layers):
        # linear layer
        M = layers[it]
        it += 1

        linear_app = M.weight.detach().cpu().numpy()

        if it < len(layers) and type(layers[it]) is BatchNorm1d:
            A = layers[it]
            var = np.diag(A.running_var.cpu().numpy())

            gamma = np.diag(A.weight.detach().cpu().numpy())
            bn = np.matmul(gamma, np.linalg.inv(var))

            linear_app = np.matmul(bn, linear_app)
            it += 1
        final_layers.append(linear_app)

    # get activations
    activations, _ = get_activations(model, X)
    activations, _ = np.unique(activations, return_inverse=True, axis=0)

    # partitions where change is not on the domain are removed.

    for i, v in enumerate(np.all(activations == activations[0, :], axis=0)):
        if v:
            activations[:, i] = 0

    # unique after corrections
    activations, _ = np.unique(activations, return_inverse=True, axis=0)

    vmin = 0. if absolute_value else -1.
    vmax = 1.

    vectors = [[] for _ in range(len(final_layers))]
    n_act = min(min_activations, len(activations))
    print_logs("n_act: %d" % n_act)
    for i in range(n_act):

        la = None
        for li, l in enumerate(final_layers):
            activated = activations[i][li * l.shape[0]: (li + 1) * l.shape[0]]

            if la is None:
                la = final_layers[li] * activated[:, np.newaxis]
            else:
                la = np.matmul(final_layers[li], la) * activated[:, np.newaxis]

            for n in la:
                vectors[li].append(n)
            continue

    return vectors, vmin, vmax


def plot_layer_output_corr_interspace(model, X, absolute_value, threshold, min_activations=10000, figure_name='ploci'):
    #Compute neural directions first
    vectors, vmin, vmax = compute_neural_directions(model, X, absolute_value, threshold, min_activations)
    plt(figure_name, figsize=(6.4, len(vectors) * 4.8))

    # compute interspace co-linearity
    for li, l in enumerate(vectors):

        cos = np.zeros((len(l), len(l)))

        for r in range(len(l)):
            for c in range(len(l)):
                if np.count_nonzero(l[r]) > 0 and np.count_nonzero(l[c]) > 0:
                    cos[r, c] = np.dot(l[r], l[c]) / np.linalg.norm(l[r]) / np.linalg.norm(l[c])
                else:
                    cos[r, c] = None
            if absolute_value:
                cos = np.abs(cos)
            if type(threshold) is not bool:
                cos = (cos > threshold).astype(int)

        plt(figure_name).subplot(len(vectors), 1, 1 + li)
        plt(figure_name).imshow(cos, vmin=vmin, vmax=vmax)
        plt(figure_name).colorbar()
        print('colorbar')
        plt(figure_name).title('Layer: '+str(li + 1))

        # plt(figure_name).tight_layout()

    return vectors


class DeriveNeuronOutput:
    def __init__(self, layer=1):
        self.activated_neurons = {}
        self.layer = layer
        self.handles = None
        self.model = None

    def __call__(self, module, input):
        print(module)
        print(input[0].size())
        input[0][0][0].backward()
        print('parameters')
        for p in self.model.parameters():
            print(p.grad)

    def register(self, model):
        self.handles = {}
        relu_count = 0
        self.model = model
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                relu_count += 1

                if self.layer is None or self.layer == relu_count:
                    handle = module.register_forward_pre_hook(self)
                    self.handles[module] = handle

    def remove(self):
        for module in list(self.handles):
            self.handles.pop(module).remove()

    def finalize(self):
        results = []
        for module in self.activated_neurons:
            results.append(np.vstack(self.activated_neurons[module]))
        return results