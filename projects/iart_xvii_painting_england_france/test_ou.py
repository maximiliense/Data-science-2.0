from numpy.random import normal
import numpy as np

from datascience.visu.util import plt, save_fig

noise = [i/10 for i in range(1, 11)]

OU = lambda x:  x - eta * x - n * normal(0, 1, 1)[0]
SGD = lambda x: x - eta * (x - n * x * np.abs(normal(0, 1, 1)[0]))

print(normal(0, 1, 1))

nb_iterations = 300
all_positions = []
for n in noise:
    pos = 150
    positions = [pos]
    eta = 0.05
    all_positions.append(positions)

    for _ in range(nb_iterations):
        pos = SGD(pos)
        positions.append(pos)


ax = plt('OU').gca()

for i, j in zip(noise, all_positions):
    print(i,j)
    ax.plot(j, label='noise: '+str(i))
ax.legend()
save_fig()
