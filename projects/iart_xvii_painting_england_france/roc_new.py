from matplotlib import cm

import pandas as pd

from datascience.visu.util import plt, save_fig, get_figure

from sklearn.metrics import roc_curve, auc, confusion_matrix

import numpy as np
import os

from engine.parameters.special_parameters import get_parameters
from engine.path import last_experiment_path

experiment_name = get_parameters('roc_experiment', 'country')
path = os.path.join(last_experiment_path(experiment_name), 'results.csv')

df = pd.read_csv(path, header='infer', sep=';')

print(df)

fpr, tpr, thresholds = roc_curve(df.true_label, df.prediction, pos_label=1)

ax = plt('roc_curve').gca()

ax.set_xlim([-0.007, 1.0])
ax.set_ylim([0.0, 1.01])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic (AUC: %.3f)' % auc(fpr, tpr))

ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random model')
ax.plot(fpr, tpr, color='blue', label='IArt')
ax.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect model')

ax.legend(loc="lower right")

ax = plt('confusion_matrix').gca()
y_threshold = (df.prediction > 0.6).astype(int)

matrix = confusion_matrix(df.true_label, y_threshold)

matrix = matrix / matrix.astype(np.float).sum(axis=1)

im = ax.imshow(matrix, cmap=cm.Greys_r, extent=(-3, 3, 3, -3))
ax.axis('off')
get_figure('confusion_matrix').colorbar(im)

save_fig()
