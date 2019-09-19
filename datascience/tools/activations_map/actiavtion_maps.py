from sklearn.model_selection import train_test_split
import numpy as np
import progressbar
import ast

from datascience.data.loader.occurrence_loader import _occurrence_loader
from datascience.data.rasters.environmental_raster_glc import PatchExtractor
from datascience.ml.neural.supervised.predict.predict_grid import predict_grid
from datascience.visu.spatial_map_plots import plot_on_map
from engine.logging import print_info
from engine.parameters import special_parameters


def plot_activations_on_map(grid_points, n_rows=3, n_cols=3, random_selection=False, selected=tuple(), log_scale=False,
                 figsize=4, mean_size=1):
        activations = np.load(special_parameters.output_path('_activations.npy'))

        # activations has shape nb points x last layer size

        plot_on_map(activations, grid_points.ids, n_cols, n_rows, figsize, log_scale, random_selection, mean_size,
                    selected=selected)


def plot_species_on_map(grid_points, _label_name, selected=0, log_scale=False,
                 figsize=5, mean_size=1):
        logits = np.load(special_parameters.output_path('_predictions.npy'))
        index = special_parameters.output_path('_index.json')

        with open(index, 'r') as f:
            s = f.read()
            index_dic = ast.literal_eval(s)

        with open(_label_name, 'r') as f:
            s = f.read()
            label_name_dic = ast.literal_eval(s)

        use_label = list(index_dic.keys())[list(index_dic.values()).index(selected)]

        true_label = index_dic[str(use_label)]
        legend = label_name_dic[true_label]

        # activations has shape nb points x last layer size

        plot_on_map(logits, grid_points.ids, 1, 1, figsize, False, False, mean_size, selected=(int(use_label),),
                    legend=(legend,),
                    output="s" + str(selected) + "_pred")


def get_species_neurons_activations(model, grid_points, batch_size=32):

        activations = predict_grid(model, grid_points, batch_size=batch_size, features_activation=True)
        predictions = predict_grid(model, grid_points, batch_size=batch_size)
        logits = predict_grid(model, grid_points, batch_size=batch_size, logit=True)

        result_path = special_parameters.output_path('_activations.npy')
        print_info("save activations:", result_path)
        np.save(result_path, activations)
        result_path = special_parameters.output_path('_predictions.npy')
        print_info("save predictions:", result_path)
        np.save(result_path, predictions)
        result_path = special_parameters.output_path('_logits.npy')
        print_info("save logits", result_path)
        np.save(result_path, logits)
        print_info("saved !")

        print_info("save weight")
        w = model.state_dict()['fc.weight']
        w = w.numpy()
        result_path = special_parameters.output_path('_weight.npy')
        np.save(result_path, w)
        print_info("saved !")


def get_species_neurons_correlations():
    activations = np.load(special_parameters.output_path('_activations.npy'))
    logits = np.load(special_parameters.output_path('_logits.npy'))
    print_info("calculate correlation matrix between features and species")

    mean_act = np.mean(activations, axis=0)
    std_act = np.std(activations, axis=0)
    norm_act = (activations - mean_act) / std_act

    mean_log = np.mean(logits, axis=0)
    std_log = np.std(logits, axis=0)
    norm_log = (logits - mean_log) / std_log

    size = activations.shape[0] * activations.shape[1]
    c = size - np.count_nonzero(activations)
    print(str(c) + "/" + str(size) + " (" + str(c * 100.0 / size) + "%)")

    matrix = np.zeros((activations.shape[1], logits.shape[1]), dtype=float)

    for i in progressbar.progressbar(range(activations.shape[0])):
        act = norm_act[i]
        log = norm_log[i]
        for j in range(norm_act.shape[1]):
            matrix[j] += (log * act[j]) / activations.shape[0]

    result_path = special_parameters.output_path('_correlation_activations.npy')
    print_info("save activations for species:", result_path)
    np.save(result_path, matrix)
    print_info("saved !")


def save_classifier_weight(model):
        w = model.state_dict()['fc.weight']
        w = w.numpy()
        print(w)
        print(type(w))
        print_info("save weight")
        result_path = special_parameters.output_path('_weight.npy')
        np.save(result_path, w)
        print_info("saved !")


def select_species_by_neuron(_label_name, grid_points, neuron, figsize=5, mean_size=1, type='correlation'):

        if type == 'correlation':
            result_path = special_parameters.output_path('_correlation_activations.npy')
            matrix = np.load(result_path)
        elif type == 'weight':
            result_path = special_parameters.output_path('_weight.npy')
            matrix = np.load(result_path)
            matrix = np.transpose(matrix)

        index = special_parameters.output_path('_index.json')

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

        with open(_label_name, 'r') as f:
            s = f.read()
            label_name_dic = ast.literal_eval(s)

        legend = []
        legend2 = matrix[neuron, test]

        for label in test:
            true_label = index_dic[str(label)]
            legend.append(label_name_dic[true_label])

        for i in range(len(legend)):
            legend[i] = "%s (%.4f)" % (legend[i], legend2[i])

        logits = np.load(special_parameters.output_path('_logits.npy'))

        activations = np.load(special_parameters.output_path('_activations.npy'))
        plot_on_map(logits, grid_points.ids, 5, 2, figsize, False, False, mean_size, selected=test, legend=legend,
                    output="n" + str(neuron) + "_species_pred")
        plot_on_map(activations, grid_points.ids, 1, 1, figsize, False, False, mean_size, selected=(neuron,),
                    output="n" + str(neuron) + "_act")


def select_neurons_by_species(_label_name, grid_points, species, figsize=5, mean_size=1, type='correlation'):

        if type == 'correlation':
            result_path = special_parameters.output_path('_correlation_activations.npy')
            matrix = np.load(result_path)
            matrix = np.transpose(matrix)
        elif type == 'weight':
            result_path = special_parameters.output_path('_weight.npy')
            matrix = np.load(result_path)

        index = special_parameters.output_path('_index.json')

        with open(index, 'r') as f:
            s = f.read()
            index_dic = ast.literal_eval(s)

        with open(_label_name, 'r') as f:
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

        logits = np.load(special_parameters.output_path('_logits.npy'))

        activations = np.load(special_parameters.output_path('_activations.npy'))
        plot_on_map(activations, grid_points.ids, 5, 2, figsize, False, False, mean_size, selected=test, legend=legend,
                    output="s" + str(true_label) + "_neurons_corr")
        plot_on_map(logits, grid_points.ids, 1, 1, figsize, False, False, mean_size, selected=(int(use_label),),
                    legend=(label_name_dic[true_label],), output="s" + str(true_label) + "_pred")


def species_train_test_occurrences(_label_name, train, val, test, species=4448):
        index = special_parameters.output_path('_index.json')

        with open(index, 'r') as f:
            s = f.read()
            index_dic = ast.literal_eval(s)

        with open(_label_name, 'r') as f:
            s = f.read()
            label_name_dic = ast.literal_eval(s)

        use_label = list(index_dic.keys())[list(index_dic.values()).index(species)]

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

        plot_on_map(r, grid_points.ids, 1, 1, 5, False, False, 1, selected=(0,), legend=(raster,),
                    output=raster)