from matplotlib import cm

from datascience.visu.util import plt, save_fig
from engine.parameters import get_parameters
from engine.path import last_experiment_path

import os
import pickle
import numpy as np

experiment_name = get_parameters('rep_experiment', 'representation')


def clean_labels(label):
    if label == 'french_painter':
        return 'French'
    elif label == 'english_painter':
        return 'English'
    else:
        return label


def plot_representation(fig_name='representation_tsne', file_name=None):
    file_name = fig_name if file_name is None else file_name
    ax = plt(fig_name, figsize=(8, 10)).gca()
    path = os.path.join(last_experiment_path(experiment_name), file_name + '.dump')
    with open(path, 'rb') as f:
        artists = pickle.load(f)
    colors = cm.tab20b(np.linspace(0, 1, len(artists)))

    for row in artists:
        ax.scatter(row[0][:, 0], row[0][:, 1], c=colors[row[1]], label=clean_labels(row[2]))

    ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt(fig_name).gcf().transFigure, prop={'size': 6},
              ncol=4, loc=3)
    ax.set_title('Painting representation')


plot_representation('representation_tsne_country')
plot_representation('representation_tsne_final_painter')
plot_representation('representation_tsne_final_country')
save_fig()
