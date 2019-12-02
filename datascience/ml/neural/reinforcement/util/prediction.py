import torch
import os

from engine.training_validation.util import construct_action, process_state, unsqueeze


def play(model_z, output_size, game_class, game_params, max_actions=5000):
    """
    Play a full game. The function will save each state, each action and the final score
    :param max_actions:
    :param model_z: to avoid infinite loops in case the model decides to come back
    :param output_size:
    :param game_class:
    :param game_params:
    :return: each state, each action and the final score
    """
    model_z.eval()
    states = []
    actions = []

    game = game_class(**game_params)
    state = game.get_state()
    states.append(state)

    xp_gpu = os.environ['QROUTING_GPU'] == 'G'

    finish = False

    c_actions = 0

    while not finish and c_actions < max_actions:
        state = process_state(state, xp_gpu)
        action = construct_action(0.01, model_z, unsqueeze(state), output_size, xp_gpu)

        action = torch.argmax(action)

        actions.append(action)
        state, _, finish = game.action(action)
        c_actions += 1
    game.save_plot('plot.jpeg')  # , old=True)
    return states, actions, game.score_
