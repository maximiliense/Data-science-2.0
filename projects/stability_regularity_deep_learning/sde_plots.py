from datascience.visu.util import plt, save_fig
import numpy as np

ax = plt('sde_omega').gca()

x = np.linspace(0, 3500, 5000)


def f(t, omega=32, eta=0.1, b=10.):
    kappa = 0.015
    kappa_k = kappa

    var_x_delta = 1.
    lambda_jk = 1.
    w_0 = 1.
    result = eta/(b * 2) * (var_x_delta/lambda_jk)
    w_t = np.log(kappa * omega * kappa_k * t + np.exp(kappa * omega * w_0))
    w_t /= (kappa*omega + (kappa ** 2) * kappa_k * (omega ** 2) * t + np.exp(kappa * omega * w_0))
    result *= w_t ** 2
    result *= (1 - np.exp(-2*lambda_jk*eta*t))
    return result


for width in [16*i for i in range(2, 5)]:
    ax.plot(x, f(x, width), label='$\\omega$=%ld' % width)
ax.legend()
save_fig('sde_omega')

ax = plt('sde_eta_b').gca()

x = np.linspace(0, 3500, 5000)

for batch_size in np.linspace(64, 512, 8):
    ax.plot(x, f(x, b=batch_size), label='$\\eta$/B=%.4lf' % (0.1/float(batch_size)))
ax.legend()
save_fig('sde_eta_b')
