from engine.util.print_colors import color, print_description_colored_title
from engine.setups.util import setup_and_run_experiment
from engine.training_validation.loss import MSELoss
from contrib.games.easy_map_3 import EasyMap
from contrib.models.cnn_tuple import Net
from util.path_finder import astar


def run(xp_name, params, epoch=0, export=False, email=0, file=False):
    params = params['game_params'] if 'game_params' in params['game_params'] else dict()

    m = EasyMap(**params)
    start = tuple(m.position)
    target = tuple(m.target)

    m.save_plot()

    map = m.game[0]
    actions = astar(map, start, target)
    for move_index in range(len(actions) - 1):
        if actions[move_index + 1][0] - actions[move_index][0] < 0:  # B
            action = 2
        elif actions[move_index + 1][0] - actions[move_index][0] > 0:  # H
            action = 3
        elif actions[move_index + 1][1] - actions[move_index][1] > 0:  # D
            action = 1
        else:
            action = 0
        m.action(action)

    m.save_plot(old=True)


def description():
    desc = print_description_colored_title('Astar resolution (Easy Map)', 'model_params')

    desc += "Game optimized by Astar"
    return desc
