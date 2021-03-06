import numpy as np
import torch

from engine.hardware import use_gpu
from engine.tensorboard import add_scalar
from engine.logging import print_info, print_warning


_memory_overflow_size = 20000


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

    if len(loader) > _memory_overflow_size and (validation_size == -1 or validation_size > _memory_overflow_size):
        print_warning(
            '[predict] The dataset size is {}. Large datasets can cause memory '
            'overflow during standard prediction...'.format(len(loader))
        )

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
            if use_gpu():
                labels = labels.cuda()
            # wrap them in Variable
            labels_variable = loss.output(labels)
            outputs = model(inputs)

            # if not test set
            if compute_loss and labels[0] != -1:
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
            print_info('Validation loss: ' + str(running_loss))
            add_scalar('Loss/Validation', running_loss)
        predictions, labels = np.asarray(y_preds), np.asarray(y_labels)

        # filtering some predicted labels
        for f in filters:
            f(predictions)

        # TODO filtering official labels

    return predictions, labels, running_loss
