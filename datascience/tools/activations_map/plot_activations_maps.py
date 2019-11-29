from sklearn.model_selection import train_test_split
import numpy as np
import ast
from datascience.data.loader.occurrence_loader import _occurrence_loader
from datascience.data.rasters.environmental_raster_glc import PatchExtractor
from datascience.visu.spatial_map_plots import plot_on_map
from engine.core import module
from engine.logging import print_info
from engine.path import output_path


@module
def plot_activations_on_map(grid_points, n_rows=3, n_cols=5, selected=tuple(), log_scale=False,
                 figsize=4, mean_size=10):
    activations = np.load(output_path('activations.npy'))

    # activations has shape nb points x last layer size

    plot_on_map(activations, grid_points.ids, n_cols=n_cols, n_rows=n_rows, figsize=figsize, log_scale=log_scale,
                mean_size=mean_size, selected=selected)


@module
def plot_species_on_map(grid_points, label_species=None, species=0, log_scale=False,
                 figsize=5, mean_size=1, softmax=False, alpha=None):
    if softmax:
        acts = np.load(output_path('predictions.npy'))
    else:
        acts = np.load(output_path('logits.npy'))

    index = output_path('index.json')

    with open(index, 'r') as f:
        s = f.read()
        index_dic = ast.literal_eval(s)

    use_label = list(index_dic.keys())[list(index_dic.values()).index(species)]

    if label_species is not None:

        with open(label_species, 'r') as f:
            s = f.read()
            label_name_dic = ast.literal_eval(s)

        true_label = index_dic[str(use_label)]
        legend = label_name_dic[true_label]

    else:
        legend = str(species)

    # activations has shape nb points x last layer size

    plot_on_map(acts, grid_points.ids, n_cols=1, n_rows=1, figsize=figsize, log_scale=log_scale,
                mean_size=mean_size, selected=(int(use_label),), alpha=alpha,
                legend=(legend,), output="s" + str(species) + "_pred")


@module
def select_species_by_neuron(grid_points, label_species, neuron, figsize=5, mean_size=1, type='correlation'):
    if type == 'correlation':
        result_path = output_path('correlation_activations.npy')
        matrix = np.load(result_path)
    elif type == 'weight':
        result_path = output_path('weight.npy')
        matrix = np.load(result_path)
        matrix = np.transpose(matrix)

    index = output_path('index.json')

    sorted = np.argsort(matrix, axis=1)
    selected = np.zeros(matrix.shape)

    for i, line in enumerate(sorted):
        for idx in line[-10:]:
            selected[i, idx] = 1.0

    selected = selected * matrix
    test = np.nonzero(selected[neuron])
    test = test[0]

    with open(index, 'r') as f:
        s = f.read()
        index_dic = ast.literal_eval(s)

    with open(label_species, 'r') as f:
        s = f.read()
        label_name_dic = ast.literal_eval(s)

    legend = []
    legend2 = matrix[neuron, test]
    legend3 = []

    for label in test:
        true_label = index_dic[str(label)]
        legend.append(label_name_dic[true_label])
        legend3.append(true_label)

    for i in range(len(legend)):
        legend[i] = "%s: %s (%.4f)" % (legend3[i], legend[i], legend2[i])

    logits = np.load(output_path('logits.npy'))

    activations = np.load(output_path('activations.npy'))
    plot_on_map(logits, grid_points.ids, n_cols=5, n_rows=2, figsize=figsize, log_scale=False,
                mean_size=mean_size, selected=test, legend=legend,
                output="n" + str(neuron) + "_species_pred")
    plot_on_map(activations, grid_points.ids, n_cols=1, n_rows=1, figsize=figsize, log_scale=False,
                mean_size=mean_size, selected=(neuron,),
                output="n" + str(neuron) + "_act")


@module
def select_neurons_by_species(grid_points, label_species, species, figsize=5, mean_size=1, type='correlation'):
    if type == 'correlation':
        result_path = output_path('correlation_activations.npy')
        matrix = np.load(result_path)
        matrix = np.transpose(matrix)
    elif type == 'weight':
        result_path = output_path('weight.npy')
        matrix = np.load(result_path)

    index = output_path('index.json')

    with open(index, 'r') as f:
        s = f.read()
        index_dic = ast.literal_eval(s)

    with open(label_species, 'r') as f:
        s = f.read()
        label_name_dic = ast.literal_eval(s)

    use_label = list(index_dic.keys())[list(index_dic.values()).index(species)]

    true_label = index_dic[str(use_label)]
    np.nan_to_num(matrix, copy=False)

    sorted = np.argsort(matrix, axis=1)
    selected = np.zeros(matrix.shape, dtype=float)
    for i, line in enumerate(sorted):
        for idx in line[-10:]:
            selected[i, idx] = 1.0
    selected = selected * matrix
    test = np.nonzero(selected[int(use_label)])
    test = test[0]

    legend = []
    legend2 = matrix[int(use_label), test]

    for n in test:
        legend.append(str(n))

    for i in range(len(legend)):
        legend[i] = "%s (%.4f)" % (legend[i], legend2[i])

    logits = np.load(output_path('logits.npy'))

    activations = np.load(output_path('activations.npy'))
    plot_on_map(activations, grid_points.ids, n_cols=5, n_rows=2, figsize=figsize, log_scale=False,
                mean_size=mean_size, selected=test, legend=legend,
                output="s" + str(true_label) + "_neurons_corr")
    plot_on_map(logits, grid_points.ids, n_cols=1, n_rows=1, figsize=figsize, log_scale=False,
                mean_size=mean_size, selected=(int(use_label),),
                legend=(label_name_dic[true_label],), output="s" + str(true_label) + "_pred")


@module
def get_correlation_csv(label_species):
    result_path = output_path('correlation_activations.npy')
    matrix = np.load(result_path)

    index = output_path('index.json')

    with open(index, 'r') as f:
        s = f.read()
        index_dic = ast.literal_eval(s)

    with open(label_species, 'r') as f:
        s = f.read()
        label_name_dic = ast.literal_eval(s)

    header = "neuron"

    for i in range(matrix.shape[1]):
        header = header+";"+label_name_dic[index_dic[str(i)]]

    neuron_index = np.arange(matrix.shape[0])

    matrix = np.insert(matrix, 0, neuron_index, axis=1)

    np.savetxt(output_path("correlations_csv.csv"), matrix, delimiter=";", header=header)
    print_info("csv file saved at "+output_path("correlations_csv.csv"))


@module
def species_train_test_occurrences(label_species, train, val, test, species=4448):
    index = output_path('index.json')

    with open(index, 'r') as f:
        s = f.read()
        index_dic = ast.literal_eval(s)

    with open(label_species, 'r') as f:
        s = f.read()
        label_name_dic = ast.literal_eval(s)

    print('label_data:', species)
    use_label = list(index_dic.keys())[list(index_dic.values()).index(species)]
    print('label_model:', use_label)

    datasets = [train, val, test]

    list_occs = [[], [], []]

    for k, d in enumerate(datasets):
        for i, label in enumerate(d.labels):
            if label == int(use_label):
                list_occs[k].append(d.dataset[i])
    print(list_occs)

    for o in list_occs[0]:
        print('%f\t%f\tcircle6\tblue\ttrain' % (o[0], o[1]))
    for o in list_occs[1]:
        print('%f\t%f\tcircle6\tgreen\tval' % (o[0], o[1]))
    for o in list_occs[2]:
        print('%f\t%f\tcircle6\tred\ttest' % (o[0], o[1]))


@module
def plot_raster(rasters, occurrences, dataset_class, validation_size=0, test_size=1, label_name='Label',
                 id_name='id', splitter=train_test_split, filters=tuple(), online_filters=tuple(),
                 postprocessing=tuple(), save_index=False, limit=None, raster="alti", **kwargs):
    _, _, grid_points = _occurrence_loader(rasters, occurrences, dataset_class, validation_size, test_size,
                                         label_name, id_name,
                                         splitter, filters, online_filters, postprocessing, save_index, limit,
                                         extractor=PatchExtractor(rasters, size=1, verbose=True), **kwargs)

    grid_points.extractor.append(raster)

    r = np.zeros((len(grid_points.dataset), 1), dtype=float)

    for i, data in enumerate(grid_points.dataset):
        value = grid_points[i][0].numpy()
        r[i, 0] = min(value[0], 2000)

    print(r)

    max_val = np.max(r)

    """

    viridis = matplotlib.cm.get_cmap('inferno', max_val)
    newcolors = viridis(np.linspace(0, 1, max_val))
    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    newcolors[1500:, :] = pink
    newcmp = ListedColormap(newcolors)
    newcmp.set_bad('grey', 1.)

    top = matplotlib.cm.get_cmap('inferno', 2000)
    bottom = matplotlib.cm.get_cmap('Blues', max_val-2000)

    newcolors = np.vstack((top(np.linspace(0, 1, 2000)),
                           bottom(np.linspace(0, 1, max_val-2000))))
    white = np.array([1, 1, 1, 1])
    newcolors[2000:, :] = white
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    newcmp.set_bad('grey', 1.)

    """

    plot_on_map(r, grid_points.ids, n_cols=1, n_rows=1, figsize=5, log_scale=False,
                mean_size=1, selected=(0,), legend=(raster,), output=raster)
