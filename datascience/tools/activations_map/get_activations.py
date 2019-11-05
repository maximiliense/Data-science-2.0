import numpy as np
import progressbar
from datascience.ml.neural.supervised.predict.predict_grid import predict_grid
from engine.core import module
from engine.logging import print_info
from engine.path import output_path


@module
def get_species_neurons_activations(model, grid_points, batch_size=32):
    activations = predict_grid(model, grid_points, batch_size=batch_size, features_activation=True)
    predictions = predict_grid(model, grid_points, batch_size=batch_size)
    logits = predict_grid(model, grid_points, batch_size=batch_size, logit=True)

    result_path = output_path('activations.npy')
    print_info("save activations:", result_path)
    np.save(result_path, activations)
    result_path = output_path('predictions.npy')
    print_info("save predictions:", result_path)
    np.save(result_path, predictions)
    result_path = output_path('logits.npy')
    print_info("save logits", result_path)
    np.save(result_path, logits)
    print_info("saved !")

    print_info("save weight")
    w = model.state_dict()['fc.weight']
    w = w.numpy()
    result_path = output_path('weight.npy')
    np.save(result_path, w)
    print_info("saved !")


@module
def get_species_neurons_correlations():
    activations = np.load(output_path('activations.npy'))
    logits = np.load(output_path('logits.npy'))
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

    result_path = output_path('correlation_activations.npy')
    print_info("save activations for species:", result_path)
    np.save(result_path, matrix)
    print_info("saved !")


def save_classifier_weight(model):
    w = model.state_dict()['fc.weight']
    w = w.numpy()
    print(w)
    print(type(w))
    print_info("save weight")
    result_path = output_path('weight.npy')
    np.save(result_path, w)
    print_info("saved !")
