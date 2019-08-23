import numpy as np
from engine.logging.logs import print_info
from datascience.ml.neural.models.util import set_model_last_layer, set_model_logit
import progressbar
import torch
from engine.parameters import special_parameters


def predict_grid(model, grid_points, batch_size=32, features_activation=False, logit=False):
    if features_activation:
        last_layer = set_model_last_layer(model, True)
    elif logit:
        last_layer = set_model_last_layer(model, False)
        training = set_model_logit(model, True)

    model.eval()

    # number of workers for the data loader
    nw = special_parameters.nb_workers
    data_loader = torch.utils.data.DataLoader(grid_points, batch_size=batch_size, num_workers=nw)

    # predictions for all points on the grid. The output is the last layer.
    print_info("get activations for all points...")
    list_activations = []
    for batch in progressbar.progressbar(data_loader):
        list_activations.append(model(batch[0]).detach().cpu().numpy())
    activations = np.concatenate(list_activations)
    # activations = np.concatenate([model(batch[0]).detach().cpu().numpy() for batch in data_loader])

    if features_activation:
        # setting back the last_layer state of the model
        set_model_last_layer(model, last_layer)
    elif logit:
        # setting back the training state of the model
        set_model_last_layer(model, last_layer)
        set_model_logit(model, training)

    return activations