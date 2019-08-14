import numpy as np
import torch

from engine.tensorboard import add_scalar
from engine.logging.logs import print_debug


def predict(model, loader, loss, export=False, filters=tuple(), validation_size=10000, compute_loss=False):
    """
        Give the prediction of the model on a test set
        :param compute_loss:
        :param filters: set some output to 0
        :param validation_size:
        :param model: the model
        :param loader: the test set loader
        :param loss: the loss function
        :param export: if False the predictions are not saved, otherwise the results are exported on file.
                       if export is true the loader must not be shuffled...
        :return: the arrays of predictions and corresponding labels
        """
    with torch.no_grad():
        total = 0
        model.eval()

        y_preds = []
        y_labels = []
        running_loss = 0.0
        idx = 0
        if hasattr(model, 'last_sigmoid') and compute_loss:
            model.last_sigmoid = False
        elif hasattr(model, 'last_sigmoid'):
            model.last_sigmoid = True

        for idx, data in enumerate(loader):

            inputs, labels = data
            # wrap them in Variable
            labels_variable = loss.output(model.p_label(labels))
            labels = model.p_label(labels)
            outputs = model(inputs)

            loss_value = loss(outputs, labels)
            running_loss += loss_value.item()
            outputs = loss.output(outputs)

            total += labels_variable.size(0)

            y_preds.extend(outputs.data.tolist())
            y_labels.extend(labels_variable.data.tolist())

            if total >= validation_size != -1 and not export:
                break
        running_loss /= (idx + 1)  # normalizing the loss
        if compute_loss:
            print_debug('Validation loss: '+str(running_loss))
            add_scalar('Loss/Validation', running_loss)
        predictions, labels = np.asarray(y_preds), np.asarray(y_labels)

        # filtering some predicted labels
        for f in filters:
            f(predictions)

        # TODO filtering official labels

    return predictions, labels, running_loss
