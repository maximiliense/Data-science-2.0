from matplotlib import cm

from datascience.visu.util import plt, save_fig, get_figure

from sklearn.metrics import roc_curve, confusion_matrix

import numpy as np

y = np.array([
    [0.8869, 1.],
    [1.-0.578, 0.],
    [0.7959, 1.],
    [0.8618, 1.],
    [1.-0.2278, 0.],
    [0.6607, 1.],
    [0.7006, 1.],
    [1.-0.4859, 0.],
    [0.6935, 1.],
    [0.9048, 1.],
    [0.6681, 1.],
    [0.7585, 1.],
    [1.-0.5063, 0.],
    [1.-0.4516, 0.],
    [1.-0.5158, 0.],
    [1.-0.5873, 0.],
    [1.-0.7682, 0.],
    [1.-0.8620, 0.],

])

fpr, tpr, thresholds = roc_curve(y[:, 1], y[:, 0], pos_label=1)

ax = plt('roc_curve').gca()

ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic example')

ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random model')
ax.plot(fpr, tpr, color='yellow', label='IArt')
ax.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect model')

ax.legend(loc="lower right")

ax = plt('confusion_matrix').gca()
y_threshold = (y > 0.6).astype(int)

matrix = confusion_matrix(y[:, 1], y_threshold[:, 0])

matrix = matrix / matrix.astype(np.float).sum(axis=1)

im = ax.imshow(matrix, cmap=cm.Greys_r, extent=(-3, 3, 3, -3))
ax.axis('off')
get_figure('confusion_matrix').colorbar(im)

save_fig()
