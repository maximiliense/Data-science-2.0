from datascience.visu.util import plt, save_fig
from numpy import linspace, power


ax = plt('x_square').gca()

x = linspace(-2, 2, 100)
x_small = linspace(0, 2, 100)
ax.set_title('$x^2$')

ax.plot(x, power(x, 2), label='$x^2+0x^3$')
ax.plot(x, power(2*x, 2), label='$4x^2+0x^3$')
ax.plot(x, power(2*x, 2)+power(x, 3)*2, label='$4x^2+2x^3$')
ax.plot(x, power(2*x, 2)+power(x, 3)*3, label='$4x^2+3x^3$')
ax.plot(x_small, 8*x_small-4, label='tangent: $8x-4$')

ax.legend()
save_fig()
