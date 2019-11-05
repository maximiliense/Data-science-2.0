import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/data/occurrences_gbif_taxref.csv', header='infer', sep=';', low_memory=False)

dataset = []
labels = []
ids = []

labels_dic = {}

for idx, row in enumerate(df.iterrows()):

    species_id = int(row[1]['Label'])
    #species_id = int(row[1]['glc19SpId'])
    labels_dic[species_id] = True
    obs_id = row[1]['id']
    #obs_id = row[1]['X_key']
    x = (row[1]['Longitude'], row[1]['Latitude'])
    ids.append(obs_id)
    dataset.append(x)
    labels.append(species_id)

x_tr, x_te, y_tr, y_te, ids_tr, ids_te = train_test_split(dataset, labels, ids, test_size=0.1, random_state=42)

#nb_labels = len(labels_dic.keys())
nb_labels = 4520
distribution = np.zeros(nb_labels, dtype=int)

for label in y_tr:
    distribution[label] += 1


distribution = np.sort(distribution)[::-1]

print(distribution)

m = distribution[0]

grad = np.arange(nb_labels)+1

plt.plot(grad, distribution, '.')



axes = plt.gca()
#axes.set_xlim(-100, nb_labels)
axes.set_xlim(-50, nb_labels)
axes.set_ylim(-10, 750)
axes.set_xlabel('species sorted by number of occurrences')
axes.set_ylabel('number of occurrences of the species')

axes.xaxis.set_major_locator(MultipleLocator(1000))
axes.xaxis.set_minor_locator(MultipleLocator(100))

axes.yaxis.set_major_locator(MultipleLocator(50))
axes.yaxis.set_minor_locator(MultipleLocator(10))

plt.savefig("nb_occurrences.png")

plt.show()





print("here")

"""
grad = (np.arange(nb_labels)/nb_labels)*100

cumul = np.zeros(nb_labels, dtype=int)

for i in range(distribution.shape[0]):
    cumul[i] = np.sum(distribution[0:i+1])

cumul = (cumul/len(y_tr))*100

plt.plot(grad, cumul, linestyle='-')

axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(0, 100)
axes.set_xlabel('% number of species')
axes.set_ylabel('% number of occurrences')

axes.xaxis.set_major_locator(MultipleLocator(10))
axes.xaxis.set_minor_locator(MultipleLocator(1))

axes.yaxis.set_major_locator(MultipleLocator(10))
axes.yaxis.set_minor_locator(MultipleLocator(1))

plt.show()
"""