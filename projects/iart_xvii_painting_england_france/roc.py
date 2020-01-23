from matplotlib import cm

from datascience.visu.util import plt, save_fig, get_figure

from sklearn.metrics import roc_curve, auc, confusion_matrix

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
    [0.8620, 1.],
    [1-0.7337, 0.],
    [0.9412, 1.],
    [1.-0.5819, 0.],
    [.2738, 1.],
    [1.-.5136, 0.],
    [.8819, 1.],
    [1.-.4387, 0.],
    [1.-.6257, 0.],
    [.7857, 1.],
    [1.-.3722, 0.],
    [1.-0.8049, 0.],
    [0.7864, 1.],
    [1.-0.2372, 0.],
    [0.7934, 1.],
    [0.9583, 1.],
    [0.9739, 1.],
    [1.-0.3556, 0.],
    [1.-0.2551, 0.],
    [1.-0.4532, 0.],
    [0.4605, 1.],
    [0.7572, 1.],
    [0.9496, 1.],
    [0.8268, 1.],
    [1.-0.4876, 0.],
    [0.8523, 1.],
    [1.-0.2629, 0.],
    [1.-0.9021, 0.],
    [0.6977, 1.],
    [0.9142, 1.],
    [1.-0.8175, 0.],
    [1.-0.4865, 0.],
    [0.9110, 1.],
    [1.-0.2159, 0.],
    [1.-0.6943, 0.],
    [1.-0.2753, 0.],
    [0.8590, 1.],
    [0.8273, 1.],
    [1.-0.5169, 0.],
    [1.-0.7412, 0.]
])

fpr, tpr, thresholds = roc_curve(y[:, 1], y[:, 0], pos_label=1)

ax = plt('roc_curve').gca()

ax.set_xlim([-0.007, 1.0])
ax.set_ylim([0.0, 1.01])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic (AUC: %.3f)' % auc(fpr, tpr))

ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random model')
ax.plot(fpr, tpr, color='yellow', label='IArt')
ax.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect model')

ax.legend(loc="lower right")

ax = plt('confusion_matrix').gca()
y_threshold = (y > 0.7).astype(int)

matrix = confusion_matrix(y[:, 1], y_threshold[:, 0])

matrix = matrix / matrix.astype(np.float).sum(axis=1)

im = ax.imshow(matrix, cmap=cm.Greys_r, extent=(-3, 3, 3, -3))
ax.axis('off')
get_figure('confusion_matrix').colorbar(im)

save_fig()
