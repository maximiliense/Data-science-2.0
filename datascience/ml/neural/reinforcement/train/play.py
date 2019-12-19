import os

import torch

from datascience.ml.neural.reinforcement.train.util import process_state, construct_action, unsqueeze


def play(model_z, output_size, game_class, game_params, nb_games):
    """
    Execute a series of game
    :param model_z:
    :param output_size:
    :param game_class:
    :param game_params:
    :param nb_games:
    :return:
    """
    total_score = 0.
    for _ in range(nb_games):
        _, _, score = _play(model_z, output_size, game_class, game_params, plot=False, max_actions=50)
        total_score += score
    return total_score / nb_games


def _play(model_z, output_size, game_class, game_params, max_actions=5000, plot=True):
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

    finish = False

    c_actions = 0

    while not finish and c_actions < max_actions:
        state = process_state(state)
        action = construct_action(0.01, model_z, unsqueeze(state), output_size)

        action = torch.argmax(action)

        actions.append(action)
        state, _, finish = game.action(action)
        c_actions += 1
    if hasattr(game, 'plot') and plot:
        game.plot()
    return states, actions, game.score_
