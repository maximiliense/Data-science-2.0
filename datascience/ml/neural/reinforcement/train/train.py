import warnings

import torch
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np

from datascience.ml.neural.reinforcement.game.util.replay_memory import ReplayMemory
from datascience.ml.neural.reinforcement.train.default_params import TRAINING_PARAMS, PREDICT_PARAMS, \
    VALIDATION_PARAMS, EXPORT_PARAMS, OPTIM_PARAMS, GAME_PARAMS
from datascience.ml.neural.reinforcement.train.play import play
from datascience.ml.neural.reinforcement.train.util import process_state, construct_action, process_state_back, \
    unsqueeze, init_game
from datascience.ml.neural.loss import load_loss, save_loss
from datascience.ml.neural.checkpoints.checkpoints import create_optimizer, save_checkpoint
from engine.hardware import use_gpu
from engine.parameters import special_parameters
from engine.path import output_path
from engine.path.path import export_epoch
from engine.util.log_email import send_email
from engine.util.log_file import save_file
from engine.logging import print_h1, print_h2, print_notification, print_errors
from engine.util.merge_dict import merge_dict_set
from engine.core import module


def _exploration(model_z, state, epsilon, game, replay_memory, output_size):
    """
    Exploring the game possibilities to learn new strategies
    :param model_z: The model that is being trained
    :param state: the current state of the game
    :param epsilon: the probability of exploration versus optimized playing
    :param game: the game
    :param replay_memory: the memory of previous games
    :return: the new state of the game after exploration and reward
    :param output_size: Size of the output of the model
    """

    model_z.eval()
    state = process_state(state)

    action = construct_action(epsilon, model_z, state, output_size)

    new_state, reward, finish = game.action(torch.argmax(action))

    replay_memory.add((process_state_back(state), action.cpu().numpy(),
                       process_state_back(new_state), reward, finish))

    return unsqueeze(new_state), reward, finish


def _optimization(model_z, batch, gamma, optimizer_, loss):
    """
    Optimizing the model for a random batch
    :param batch: the current batch that will be used to optimize the model
    :param model_z: the model to train
    :param gamma: the gamma value for the Q function
    :param optimizer_: the optimizer
    :param loss: the loss criterion
    :return: the loss value
    """
    model_z.train()

    # processing mini batches
    state_batch, action_batch, state_1_batch, reward_batch, finished_batch = batch

    state_batch = process_state(state_batch)

    state_1_batch = process_state(state_1_batch)

    # eventually combine state and reward so that the model can train on both

    if use_gpu():  # put on GPU if the user asked it
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        finished_batch = finished_batch.cuda()
    output_1_batch = model_z(*state_1_batch)

    finished = finished_batch.unsqueeze(1).float()
    reward_batch = reward_batch.unsqueeze(1).float()

    # Bellman equation
    y_batch = reward_batch + gamma * torch.max(torch.mul(output_1_batch, 1 - finished), dim=1)[0].unsqueeze(1)
    y_batch = y_batch.detach()  # y_batch is not trainable as it is the target

    output = model_z(*state_batch)

    # if state and reward are combined to have more complex training procedures,
    # this line should be replaced with a loss function
    q_value = torch.sum(output * action_batch.float(), dim=1).unsqueeze(1)

    optimizer_.zero_grad()
    loss_value = loss(q_value, y_batch)

    loss_value.backward()
    optimizer_.step()

    return loss_value.item()


@module
def fit(model_z, game_class, game_params=None, training_params=None, predict_params=None, validation_params=None,
        export_params=None, optim_params=None):
    """
    This function is the core of an experiment. It performs the ml procedure as well as the call to validation.
    :param game_params:
    :param game_class:
    :param training_params: parameters for the training procedure
    :param optim_params:
    :param export_params:
    :param validation_params:
    :param predict_params:
    :param model_z: the model that should be trained
    """
    # configuration
    game_params, training_params, predict_params, validation_params, export_params, optim_params = merge_dict_set(
        game_params, GAME_PARAMS,
        training_params, TRAINING_PARAMS,
        predict_params, PREDICT_PARAMS,
        validation_params, VALIDATION_PARAMS,
        export_params, EXPORT_PARAMS,
        optim_params, OPTIM_PARAMS
    )

    validation_path = output_path('validation.txt')

    output_size = model_z.output_size if hasattr(model_z, 'output_size') else model_z.module.output_size

    # training parameters
    optim = optim_params.pop('optimizer')
    iterations = training_params.pop('iterations')
    gamma = training_params.pop('gamma')
    batch_size = training_params.pop('batch_size')
    loss = training_params.pop('loss')
    log_modulo = training_params.pop('log_modulo')
    val_modulo = training_params.pop('val_modulo')
    first_epoch = training_params.pop('first_epoch')
    rm_size = training_params.pop('rm_size')
    epsilon_start = training_params.pop('epsilon_start')
    epsilon_end = training_params.pop('epsilon_end')

    validate = special_parameters.evaluate
    export = special_parameters.export
    do_train = special_parameters.train
    max_iterations = max(iterations)

    game = game_class(**game_params)

    replay_memory = ReplayMemory(rm_size)

    if do_train and first_epoch < max(iterations):
        print_h1('Training: ' + special_parameters.setup_name)

        state = unsqueeze(init_game(game, replay_memory, output_size, len(replay_memory)))
        memory_loader = torch.utils.data.DataLoader(
            replay_memory, shuffle=True, batch_size=batch_size,
            num_workers=16, drop_last=True
        )

        if batch_size > len(replay_memory):
            print_errors('Batch size is bigger than available memory...', do_exit=True)

        loss_logs = [] if first_epoch < 1 else load_loss('train_loss')

        loss_val_logs = [] if first_epoch < 1 else load_loss('validation_loss')

        rewards_logs = [] if first_epoch < 1 else load_loss('train_rewards')
        rewards_val_logs = [] if first_epoch < 1 else load_loss('val_rewards')

        epsilon_decrements = np.linspace(epsilon_start, epsilon_end, iterations[-1])

        opt = create_optimizer(model_z.parameters(), optim, optim_params)

        scheduler = MultiStepLR(opt, milestones=list(iterations), gamma=gamma)

        # number of batches in the ml
        epoch_size = len(replay_memory)

        # one log per epoch if value is -1
        log_modulo = epoch_size if log_modulo == -1 else log_modulo

        epoch = 0

        running_loss = 0.0
        running_reward = 0.0
        norm_opt = 0
        norm_exp = 0

        for epoch in range(max_iterations):

            if epoch < first_epoch:
                # opt.step()
                _skip_step(scheduler, epoch)
                continue
            # saving epoch to enable restart
            export_epoch(epoch)

            epsilon = epsilon_decrements[epoch]

            model_z.train()

            # printing new epoch
            print_h2('-' * 5 + ' Epoch ' + str(epoch + 1) + '/' + str(max_iterations) +
                     ' (lr: ' + str(scheduler.get_lr()) + ') ' + '-' * 5)

            for idx, data in enumerate(memory_loader):

                # the two Q-learning steps
                state, _, finish = _exploration(model_z, state, epsilon, game, replay_memory, output_size)

                if finish:
                    # if the game is finished, we save the score
                    running_reward += game.score_
                    norm_exp += 1
                # zero the parameter gradients

                running_loss += _optimization(model_z, data, gamma, opt, loss)
                norm_opt += 1

            if epoch % log_modulo == log_modulo - 1:
                print('[%d, %5d]\tloss: %.5f' % (epoch + 1, idx + 1, running_loss / log_modulo))
                print('\t\t reward: %.5f' % (running_reward / log_modulo))
                loss_logs.append(running_loss / log_modulo)
                rewards_logs.append(running_reward / log_modulo)
                running_loss = 0.0
                running_reward = 0.0
                norm_opt = 0
                norm_exp = 0

            # end of epoch update of learning rate scheduler
            scheduler.step(epoch + 1)

            # saving the model and the current loss after each epoch
            save_checkpoint(model_z, optimizer=opt)

            # validation of the model
            if epoch % val_modulo == val_modulo - 1:
                validation_id = str(int((epoch + 1) / val_modulo))

                # validation call
                loss_val = play(model_z, output_size, game_class, game_params, 2)

                loss_val_logs.append(loss_val)

                res = '\n[validation_id:' + validation_id + ']\n' + str(loss_val)

                print_notification(res)

                if special_parameters.mail == 2:
                    send_email('Results for XP ' + special_parameters.setup_name + ' (epoch: ' + str(epoch + 1) + ')',
                               res)
                if special_parameters.file:
                    save_file(validation_path, 'Results for XP ' + special_parameters.setup_name +
                              ' (epoch: ' + str(epoch + 1) + ')', res)

                # checkpoint
                save_checkpoint(model_z, optimizer=opt, validation_id=validation_id)

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

    # final validation
    print_h1('Validation/Export: ' + special_parameters.setup_name)

    loss_val = play(model_z, output_size, game_class, game_params, 500)

    res = '' + loss_val

    print_notification(res, end='')

    if special_parameters.mail >= 1:
        send_email('Final results for XP ' + special_parameters.setup_name, res)
    if special_parameters.file:
        save_file(validation_path, 'Final results for XP ' + special_parameters.setup_name, res)


def _skip_step(lr_scheduler, epoch):
    warnings.filterwarnings("ignore")
    lr_scheduler.step(epoch + 1)
    warnings.filterwarnings("default")
