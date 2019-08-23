import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import random
import numpy as np
from engine.logging import print_info
from datascience.visu.util import plt, get_figure, save_fig


def plot_on_map(activations, map_ids, n_cols, n_rows, figsize, log_scale, random_selection, mean_size, selected=tuple(),
                output="activations"):
    if log_scale:
        activations = activations + 1.0
        activations = np.log(activations)

    print_info("construct array activation map...")
    pos = []
    max_x = 0
    max_y = 0
    for id_ in map_ids:
        x, y = id_.split("_")
        x, y = int(x), int(y)
        pos.append((x, y))
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    size = max(max_x + 1, max_y + 1)
    while size % mean_size != 0:
        size += 1
    nb = n_cols * n_rows
    act_map = np.ndarray((nb, size, size))
    act_map[:] = np.nan

    print_info("select neurons to print...")
    if random_selection:
        list_select = random.sample(list(range(activations.shape[1])), nb)
    elif len(selected) > 0:
        list_select = selected
    else:
        list_select = list(range(nb))

    print_info("fill activation map array...")
    for k, act in enumerate(activations):
        for idx, j in enumerate(list_select):
            x, y = pos[k][0], pos[k][1]
            act_map[idx, x, y] = act[j]

    """
    fig, axs = plt.subplots(n_rows, n_cols, sharex='col', sharey='row',
                            figsize=(n_cols*figsize*1.2, n_rows*figsize))
    fig.subplots_adjust(wspace=0.5)
    plt.tight_layout(pad=1.5)

    print_logs("make figure...")
    for j in range(nb):
        if mean_size != 1:
            height, width = act_map[j].shape
            act_map_j = np.ma.average(np.split(np.ma.average(np.split(act_map[j], width // mean_size, axis=1),
                                               axis=2), height // mean_size, axis=1), axis=2)
        else:
            act_map_j = act_map[j]

        masked_array = np.ma.array(act_map_j, mask=np.isnan(act_map_j))
        cmap = matplotlib.cm.inferno
        cmap.set_bad('grey', 1.)
        im = axs[j // n_cols, j % n_cols].imshow(masked_array, cmap=cmap, interpolation='none')
        axs[j // n_cols, j % n_cols].set_title(str(list_select[j]))
        divider = make_axes_locatable(axs[j // n_cols, j % n_cols])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    """

    plt(output, figsize=(n_cols * figsize * 1.2, n_rows * figsize))
    fig = get_figure(output)
    fig.subplots_adjust(wspace=0.05)
    fig.tight_layout(pad=0.05)

    print_info("make figure...")
    for j in range(nb):
        if mean_size != 1:
            height, width = act_map[j].shape
            act_map_j = np.ma.average(np.split(np.ma.average(np.split(act_map[j], width // mean_size, axis=1),
                                                             axis=2), height // mean_size, axis=1), axis=2)
        else:
            act_map_j = act_map[j]

        masked_array = np.ma.array(act_map_j, mask=np.isnan(act_map_j))
        cmap = matplotlib.cm.inferno
        cmap.set_bad('grey', 1.)
        ax = plt(output).subplot(n_rows, n_cols, j + 1)
        im = plt(output).imshow(masked_array, cmap=cmap, interpolation='none')
        plt(output).title(str(list_select[j]), fontsize=24)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt(output).colorbar(im, cax=cax)

    save_fig(figure_name=output, extension='png')