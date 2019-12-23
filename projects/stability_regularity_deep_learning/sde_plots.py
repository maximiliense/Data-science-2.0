from datascience.visu.util import plt, save_fig
import numpy as np

ax = plt('sde_kappa').gca()

x = np.linspace(0, 3500, 5000)


def f(t, kappa=0.2, lambda_jk=1., b=10.):
    eta = 0.001
    var_x = 1.
    w_0 = 1.
    result = (eta / b) * (var_x/lambda_jk)
    result *= (1/(kappa**2) * np.log(kappa ** 2 * t + np.exp(kappa*w_0))/(1+kappa ** 2 * t + np.exp(kappa * w_0))) ** 2
    result *= (1 - np.exp(-2*lambda_jk*eta*t))
    return result


for k in np.linspace(0.1, 0.2, 10):
    ax.plot(x, f(x, k), label='%.2f' % k)
ax.legend()
save_fig('sde_kappa')

ax = plt('sde_lambda').gca()

x = np.linspace(0, 3500, 5000)

for la in np.linspace(0.5, 10, 5):
    ax.plot(x, f(x, lambda_jk=la), label='%.2f' % la)
ax.legend()
save_fig('sde_lambda')
