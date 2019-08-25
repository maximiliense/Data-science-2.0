import torch
import torch.optim as optimizer
from torch.optim.lr_scheduler import MultiStepLR

from datascience.ml.neural.models.util import save_model
from datascience.ml.neural.supervised.callbacks import init_callbacks, run_callbacks, finish_callbacks
from datascience.ml.neural.loss import CELoss, load_loss, save_loss
from datascience.ml.neural.supervised.predict import predict
from datascience.ml.evaluation import validate, export_results
from engine.parameters import special_parameters
from engine.path import output_path
from engine.path.path import export_epoch
from engine.util.log_email import send_email
from engine.util.log_file import save_file
from engine.logging import print_errors, print_h1, print_info, print_h2, print_notification
from engine.util.merge_dict import merge_smooth
from engine.tensorboard import add_scalar
from engine.core import module


@module
def fit(model_z, train, test, val=None, training_params=None, predict_params=None, validation_params=None,
        export_params=None, optim_params=None):
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
    """
    # configuration
    training_params, predict_params, validation_params, export_params, optim_params = configure(
        training_params, predict_params, validation_params, export_params, optim_params
    )

    train_loader, test_loader, val_loader = dataset_setup(train, test, val, **training_params)

    validation_path = output_path('validation.txt')

    # training parameters
    optim = optim_params.pop('optimizer')
    lr = training_params.pop('lr')
    iterations = training_params.pop('iterations')
    gamma = training_params.pop('gamma')
    loss = training_params.pop('loss')
    log_modulo = training_params.pop('log_modulo')
    val_modulo = training_params.pop('val_modulo')
    first_epoch = training_params.pop('first_epoch')

    # callbacks for ml tests
    vcallback = validation_params.pop('vcallback') if 'vcallback' in validation_params else None

    # before ml callback
    if vcallback is not None and not (special_parameters.validation_only or
                                      special_parameters.export or len(iterations) == 0):
        init_callbacks(vcallback, val_modulo, max(iterations) // val_modulo, train_loader.dataset, model_z)
    validation_only = special_parameters.validation_only
    export = special_parameters.export
    if not (validation_only or export or len(iterations) == 0 or train_loader is None):
        max_iterations = max(iterations)
        if first_epoch >= max_iterations:
            print_errors('you can\'t start from epoch ' + str(first_epoch + 1))
            exit()

        print_h1('Training: ' + special_parameters.setup_name)

        model_path = output_path('models/model.torch')

        loss_logs = [] if first_epoch < 1 else load_loss('train_loss')

        loss_val_logs = [] if first_epoch < 1 else load_loss('validation_loss')

        sgd = optim(model_z.parameters(), lr=lr, **optim_params)
        scheduler = MultiStepLR(sgd, milestones=list(iterations), gamma=gamma)

        # number of batches in the ml
        epoch_size = len(train_loader)

        # one log per epoch if value is -1
        log_modulo = epoch_size if log_modulo == -1 else log_modulo
        for epoch in range(max_iterations):
            if epoch < first_epoch:
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
                labels = model_z.p_label(labels)

                # zero the parameter gradients
                sgd.zero_grad()
                outputs = model_z(inputs)
                loss_value = loss(outputs, labels)
                loss_value.backward()

                sgd.step()

                # print math
                running_loss += loss_value.item()
                if idx % log_modulo == log_modulo - 1:  # print every log_modulo mini-batches
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, idx + 1, running_loss / log_modulo))

                    # tensorboard support
                    add_scalar('Loss/train', running_loss / log_modulo)
                    loss_logs.append(running_loss / log_modulo)
                    running_loss = 0.0
            # end of epoch update of learning rate scheduler
            scheduler.step()
            # train_loader.data.reverse = not train_loader.data.reverse  # This is to check
            # the oscillating loss probably due to SGD and momentum...

            # saving the model and the current loss after each epoch
            save_model(model_z, model_path)

            # validation of the model
            if epoch % val_modulo == val_modulo - 1:
                validation_id = str(int((epoch + 1) / val_modulo))

                # validation call
                predictions, labels, loss_val = predict(
                    model_z, val_loader, loss, **predict_params, compute_loss=True
                )
                loss_val_logs.append(loss_val)

                res = '\n[validation_id:' + validation_id + ']\n' + validate(predictions, labels, **validation_params)

                print_notification(res)

                if special_parameters.mail == 2:
                    send_email('Results for XP ' + special_parameters.setup_name + ' (epoch: ' + str(epoch + 1) + ')',
                               res)
                if special_parameters.file:
                    save_file(validation_path, 'Results for XP ' + special_parameters.setup_name +
                              ' (epoch: ' + str(epoch + 1) + ')', res)

                # save model used for validation
                mvp = output_path('models/model.torch', validation_id=validation_id)
                save_model(model_z, mvp)

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

    # final validation
    print_h1('Validation/Export: ' + special_parameters.setup_name)

    predictions, labels, val_loss = predict(model_z, test_loader, loss, validation_size=-1, **predict_params)

    if special_parameters.validation_only or not special_parameters.export:

        res = validate(predictions, labels, **validation_params, final=True)

        print_notification(res, end='')

        if special_parameters.mail >= 1:
            send_email('Final results for XP ' + special_parameters.setup_name, res)
        if special_parameters.file:
            save_file(validation_path, 'Final results for XP ' + special_parameters.setup_name, res)
        # callback
        if vcallback is not None and not (validation_only or export or len(iterations) == 0):
            finish_callbacks(vcallback)
    if special_parameters.export:
        export_results(test_loader.dataset, predictions, **export_params)

    return predictions


def configure(training_params, predict_params, validation_params, export_params, optim_params):
    """
    configure default parameters
    :param training_params:
    :param predict_params:
    :param validation_params:
    :param export_params:
    :param optim_params:
    :return:
    """
    training_params = {} if training_params is None else training_params
    merge_smooth(
        training_params,
        {
            'batch_size': 32,
            'lr': 0.1,
            'iterations': None,
            'gamma': 0.1,
            'loss': CELoss(),
            'val_modulo': 1,
            'log_modulo': -1,
            'first_epoch': special_parameters.first_epoch
        }
    )

    predict_params = {} if predict_params is None else predict_params
    validation_params = {} if validation_params is None else validation_params
    merge_smooth(validation_params, {'metrics': tuple()})

    export_params = {} if export_params is None else export_params

    optim_params = {} if optim_params is None else optim_params
    merge_smooth(optim_params, {'momentum': 0.9, 'weight_decay': 0, 'optimizer': optimizer.SGD})

    return training_params, predict_params, validation_params, export_params, optim_params


def dataset_setup(train, test, val=None, batch_size=32, bs_test=None, train_shuffle=True, test_shuffle=False, **kwargs):
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
