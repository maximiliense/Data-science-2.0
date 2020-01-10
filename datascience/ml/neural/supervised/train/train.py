import warnings

import torch
from torch.optim.lr_scheduler import MultiStepLR

from datascience.ml.metrics.learning_statistics import Statistics
from datascience.ml.neural.supervised.callbacks import init_callbacks, run_callbacks, finish_callbacks
from datascience.ml.neural.loss import load_loss, save_loss
from datascience.ml.neural.supervised.predict import predict
from datascience.ml.evaluation import validate, export_results
from datascience.ml.neural.checkpoints.checkpoints import create_optimizer, save_checkpoint
from datascience.ml.neural.supervised.train.default_params import TRAINING_PARAMS, OPTIM_PARAMS, EXPORT_PARAMS, \
    VALIDATION_PARAMS, PREDICT_PARAMS, MODEL_SELECTION_PARAMS
from engine.hardware import use_gpu
from engine.parameters import special_parameters
from engine.path import output_path
from engine.path.path import export_epoch
from engine.util.log_email import send_email
from engine.util.log_file import save_file
from engine.logging import print_h1, print_h2, print_notification, print_errors
from engine.util.merge_dict import merge_dict_set
from engine.tensorboard import add_scalar
from engine.core import module


@module
def fit(model_z, train, test, val=None, training_params=None, predict_params=None, validation_params=None,
        export_params=None, optim_params=None, model_selection_params=None):
    """
    This function is the core of an experiment. It performs the ml procedure as well as the call to validation.
    :param training_params: parameters for the training procedure
    :param val: validation set
    :param test: the test set
    :param train: The training set
    :param optim_params:
    :param export_params:
    :param validation_params:
    :param predict_params:
    :param model_z: the model that should be trained
    :param model_selection_params:
    """
    # configuration

    training_params, predict_params, validation_params, export_params, optim_params, \
        cv_params = merge_dict_set(
            training_params, TRAINING_PARAMS,
            predict_params, PREDICT_PARAMS,
            validation_params, VALIDATION_PARAMS,
            export_params, EXPORT_PARAMS,
            optim_params, OPTIM_PARAMS,
            model_selection_params, MODEL_SELECTION_PARAMS
        )

    train_loader, test_loader, val_loader = _dataset_setup(train, test, val, **training_params)

    statistics_path = output_path('metric_statistics.dump')

    metrics_stats = Statistics(model_z, statistics_path, **cv_params) if cv_params.pop('cross_validation') else None

    validation_path = output_path('validation.txt')

    # training parameters
    optim = optim_params.pop('optimizer')
    iterations = training_params.pop('iterations')
    gamma = training_params.pop('gamma')
    loss = training_params.pop('loss')
    log_modulo = training_params.pop('log_modulo')
    val_modulo = training_params.pop('val_modulo')
    first_epoch = training_params.pop('first_epoch')

    # callbacks for ml tests
    vcallback = validation_params.pop('vcallback') if 'vcallback' in validation_params else None

    if iterations is None:
        print_errors('Iterations must be set', exception=TrainingConfigurationException('Iterations is None'))

    # before ml callback
    if vcallback is not None and special_parameters.train and first_epoch < max(iterations):
        init_callbacks(vcallback, val_modulo, max(iterations) // val_modulo, train_loader.dataset, model_z)

    max_iterations = max(iterations)

    if special_parameters.train and first_epoch < max(iterations):
        print_h1('Training: ' + special_parameters.setup_name)

        loss_logs = [] if first_epoch < 1 else load_loss('loss_train')

        loss_val_logs = [] if first_epoch < 1 else load_loss('loss_validation')

        opt = create_optimizer(model_z.parameters(), optim, optim_params)

        scheduler = MultiStepLR(opt, milestones=list(iterations), gamma=gamma)

        # number of batches in the ml
        epoch_size = len(train_loader)

        # one log per epoch if value is -1
        log_modulo = epoch_size if log_modulo == -1 else log_modulo

        epoch = 0
        for epoch in range(max_iterations):

            if epoch < first_epoch:
                # opt.step()
                _skip_step(scheduler, epoch)
                continue
            # saving epoch to enable restart
            export_epoch(epoch)
            model_z.train()

            # printing new epoch
            print_h2('-' * 5 + ' Epoch ' + str(epoch + 1) + '/' + str(max_iterations) +
                     ' (lr: ' + str(scheduler.get_lr()) + ') ' + '-' * 5)

            running_loss = 0.0

            for idx, data in enumerate(train_loader):

                # get the inputs
                inputs, labels = data

                # wrap labels in Variable as input is managed through a decorator
                # labels = model_z.p_label(labels)
                if use_gpu():
                    labels = labels.cuda()

                # zero the parameter gradients
                opt.zero_grad()
                outputs = model_z(inputs)
                loss_value = loss(outputs, labels)
                loss_value.backward()

                opt.step()

                # print math
                running_loss += loss_value.item()
                if idx % log_modulo == log_modulo - 1:  # print every log_modulo mini-batches
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, idx + 1, running_loss / log_modulo))

                    # tensorboard support
                    add_scalar('Loss/train', running_loss / log_modulo)
                    loss_logs.append(running_loss / log_modulo)
                    running_loss = 0.0

            # end of epoch update of learning rate scheduler
            scheduler.step(epoch + 1)

            # saving the model and the current loss after each epoch
            save_checkpoint(model_z, optimizer=opt)

            # validation of the model
            if epoch % val_modulo == val_modulo - 1:
                validation_id = str(int((epoch + 1) / val_modulo))

                # validation call
                predictions, labels, loss_val = predict(
                    model_z, val_loader, loss, **predict_params, compute_loss=True
                )
                loss_val_logs.append(loss_val)

                res = '\n[validation_id:' + validation_id + ']\n' + validate(
                    predictions, labels, validation_id=validation_id, statistics=metrics_stats, **validation_params
                )

                # save statistics for robust cross validation
                if metrics_stats:
                    metrics_stats.save()

                print_notification(res)

                if special_parameters.mail == 2:
                    send_email('Results for XP ' + special_parameters.setup_name + ' (epoch: ' + str(epoch + 1) + ')',
                               res)
                if special_parameters.file:
                    save_file(validation_path, 'Results for XP ' + special_parameters.setup_name +
                              ' (epoch: ' + str(epoch + 1) + ')', res)

                # checkpoint
                save_checkpoint(model_z, optimizer=opt, validation_id=validation_id)

                # callback
                if vcallback is not None:
                    run_callbacks(vcallback, (epoch + 1) // val_modulo)

            # save loss
            save_loss(
                {  # // log_modulo * log_modulo in case log_modulo does not divide epoch_size
                    'train': (loss_logs, log_modulo),
                    'validation': (loss_val_logs, epoch_size // log_modulo * log_modulo * val_modulo)
                },
                ylabel=str(loss)
            )

        # saving last epoch
        export_epoch(epoch + 1)  # if --restart is set, the train will not be executed

    # callback
    if vcallback is not None and not special_parameters.train:
        finish_callbacks(vcallback)

    # final validation
    if special_parameters.evaluate or special_parameters.export:
        print_h1('Validation/Export: ' + special_parameters.setup_name)
        if metrics_stats is not None:
            # change the parameter states of the model to best model
            metrics_stats.switch_to_best_model()

        predictions, labels, val_loss = predict(model_z, test_loader, loss, validation_size=-1, **predict_params)

        if special_parameters.evaluate:

            res = validate(predictions, labels, statistics=metrics_stats, **validation_params, final=True)

            print_notification(res, end='')

            if special_parameters.mail >= 1:
                send_email('Final results for XP ' + special_parameters.setup_name, res)
            if special_parameters.file:
                save_file(validation_path, 'Final results for XP ' + special_parameters.setup_name, res)

        if special_parameters.export:
            export_results(test_loader.dataset, predictions, **export_params)

    return metrics_stats


def _dataset_setup(train, test, val=None, batch_size=32, bs_test=None,
                   train_shuffle=True, test_shuffle=False, **kwargs):
    # ignore kwargs
    locals().update(kwargs)
    if val is None:
        val = test
    if type(train) is not torch.utils.data.DataLoader:
        if batch_size == -1:
            bs_test = min(len(test), len(val))
            batch_size = len(train)

        if bs_test is None:
            bs_test = batch_size

        num_workers = special_parameters.nb_workers
        if len(test) != 0:
            test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=bs_test, num_workers=num_workers)
        else:
            test_loader = None
        if len(train) != 0:
            train_loader = torch.utils.data.DataLoader(train, shuffle=train_shuffle, batch_size=batch_size,
                                                       num_workers=num_workers)
        else:
            train_loader = None
        if len(val) != 0:
            val_loader = torch.utils.data.DataLoader(val, shuffle=test_shuffle, batch_size=bs_test,
                                                     num_workers=num_workers)
        else:
            val_loader = None

        return train_loader, test_loader, val_loader
    else:
        return train, test, val


def _skip_step(lr_scheduler, epoch):
    warnings.filterwarnings("ignore")
    lr_scheduler.step(epoch + 1)
    warnings.filterwarnings("default")


class TrainingConfigurationException(Exception):
    def __init__(self, message):
        super(TrainingConfigurationException, self).__init__(message)
