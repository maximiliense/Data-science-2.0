import os
import numpy as np

import torch
import torch.optim as optimizer
from torch.optim.lr_scheduler import MultiStepLR

from engine.util.print_colors import color
from engine.training_validation.logs import validation_logs, epoch_title, training_logs
from engine.training_validation.util import construct_action, init_game, process_state, process_state_back, unsqueeze
from engine.util.plot_curve import plot_curve

xp_gpu = False


def exploration(model_z, state, epsilon, game, replay_memory, output_size):
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
    state = process_state(state, xp_gpu)

    action = construct_action(epsilon, model_z, state, output_size, xp_gpu)

    new_state, reward, finish = game.action(torch.argmax(action))

    replay_memory.add((process_state_back(state), action.cpu().numpy(),
                       process_state_back(new_state), reward, finish))

    return unsqueeze(new_state), reward, finish


def optimization(model_z, batch, gamma, optimizer_, loss):
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

    state_batch = process_state(state_batch, xp_gpu)

    state_1_batch = process_state(state_1_batch, xp_gpu)

    # eventually combine state and reward so that the model can train on both

    if xp_gpu:  # put on GPU if the user asked it
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


def train(iterations, lr, lr_decay, first_epoch, model_z, xp_name, loss, game_class, game_params, replay_memory,
          batch_size, log_modulo=10, validation_modulo=10, play_only=False, send_results=0, save_results=False,
          weight_decay=0, momentum=0.9, epsilon_start=0.9, epsilon_end=0.1, gamma=0.5, save_validation_model=False):
    # if complex training procedures are added, the loss should be a parameter

    # checking iterations and first epoch
    if first_epoch >= max(iterations):
        print('you can\'t start from epoch ' + str(first_epoch + 1))
        exit()

    # configuring GPU usage
    global xp_gpu
    xp_gpu = os.environ['QROUTING_GPU'] == 'G'

    nb_validation_game = 1

    # setting up the root directory
    root_dir = os.environ['QROUTING_ROOT_DIR']

    validation_txt_path = root_dir + '/' + xp_name + '_validation.txt'

    loss_logs = []
    rewards_logs = []
    val_logs = []

    # various training files will be exported at the following path
    model_path = root_dir + '/' + xp_name + '_model.torch'
    loss_path = root_dir + '/' + xp_name + '_loss.png'
    rewards_path = root_dir + '/' + xp_name + '_rewards.png'
    validation_path = root_dir + '/' + xp_name + '_val.png'

    if hasattr(model_z, 'output_size'):
        output_size = model_z.output_size
    else:
        output_size = model_z.module.output_size

    if not play_only:
        sgd = optimizer.SGD(model_z.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = MultiStepLR(sgd, milestones=iterations, gamma=lr_decay)

        game = game_class(**game_params)

        state = unsqueeze(init_game(game, replay_memory, output_size, len(replay_memory)))

        memory_loader = torch.utils.data.DataLoader(replay_memory, shuffle=True, batch_size=batch_size,
                                                    num_workers=16, drop_last=True)
        if batch_size > len(replay_memory):
            print(color.RED + 'Batch size is bigger than the available memory...' + color.END)
            exit()
        epsilon_decrements = np.linspace(epsilon_start, epsilon_end, iterations[-1])

        # epoch_size = int(replay_memory.replay_memory_size / batch_size)
        running_loss = 0.0
        running_reward = 0.0
        norm_opt = 0
        norm_exp = 0
        print(color.GREEN + '\n' + '*' * 10 + ' Training: ' + xp_name + ' ' + '*' * 10 + '\n' + color.END)
        for epoch in range(max(iterations)):
            scheduler.step()

            if epoch < first_epoch:
                continue
            epsilon = epsilon_decrements[epoch]
            if epoch % log_modulo == 0:
                # printing information about current epoch

                print(epoch_title(log_modulo, epoch, epsilon, scheduler))

            # print('New epoch')

            for idx, batch in enumerate(memory_loader):

                # the two Q-learning steps
                state, _, finish = exploration(model_z, state, epsilon, game, replay_memory, output_size)

                if finish:
                    # if the game is finished, we save the score
                    running_reward += game.score_
                    norm_exp += 1

                running_loss += optimization(model_z, batch, gamma, sgd, loss)
                norm_opt += 1

            # printing values about the loss and the rewards
            if epoch % log_modulo == log_modulo - 1:
                training_logs(norm_exp, norm_opt, running_loss, running_reward, loss_logs, rewards_logs, epoch,
                              model_z, model_path)

                plot_curve(loss_logs, loss_path, 'Loss', 'x' + str(log_modulo) + ' batches', 'Loss metric')

                plot_curve(rewards_logs, rewards_path, 'Rewards', 'x' + str(log_modulo) + ' batches', 'Average rewards')
                # resetting loss and rewards
                running_loss = 0.0
                running_reward = 0.0
                norm_opt = 0
                norm_exp = 0

            # printing charts, saving model, etc.
            if epoch % validation_modulo == validation_modulo - 1:
                validation_logs(epoch, validation_modulo, val_logs, send_results, save_results, model_z, output_size,
                                game_class, game_params, nb_validation_game, xp_name, validation_txt_path, root_dir,
                                save_validation_model=save_validation_model)

                plot_curve(val_logs, validation_path, 'Validation rewards', 'x' + str(log_modulo) + ' batches',
                           'Average rewards')

    # final validation
    desc = color.GREEN + '\n' + '*' * 10 + ' Final validation: ' + xp_name + ' ' + '*' * 10 + '\n' + color.END
    print(desc)
    validation_logs(-1, None, val_logs, send_results, save_results, model_z, output_size, game_class,
                    game_params, nb_validation_game, xp_name, validation_txt_path, root_dir, final=True)
