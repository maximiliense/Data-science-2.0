import numpy as np

from datascience.visu.util import plt, save_fig

eta = (0.5, 0.1)
B = (4, 8, 16, 32)
rho = 0.01


def pred_error(time, _eta, _B):

    return 1.-np.exp(
        -_eta/(2*_B)*rho/(rho**2*time+(_eta/(2*_B)+0.1)**(1/rho)))


ax = plt('sim_loss').gca()

for e in eta:
    for b in B:
        loss = []
        for i in range(80):
            loss.append(pred_error(i, e, b))

        ax.plot(loss, label="$\\eta={}, B={}$".format(e, b))
        ax.legend()

save_fig()
