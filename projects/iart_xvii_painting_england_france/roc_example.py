from sklearn.metrics import auc

from datascience.visu.util import plt, save_fig

fpr = [0., 0.2, 0.5, 0.85, 1.]
tpr = [0., 0.7, 0.9, 0.97, 1.]


ax = plt('roc_curve').gca()

ax.set_xlim([-0.007, 1.0])
ax.set_ylim([0.0, 1.01])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic')

ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random model (AUC = 0.5)')
ax.plot(fpr, tpr, color='blue', label='Some model (AUC = %.1f)' % auc(fpr, tpr))
ax.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', label='Perfect model (AUC = 1.0)')

plt('roc_curve').gcf().legend(loc='center right')

save_fig()
