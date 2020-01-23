from engine.util.print_colors import color, print_description_colored_title
from contrib.games.offshore_regatta import OffshoreRegatta
from util.astar_regatta import astar


def run(xp_name, params, **kwargs):
    params = params['game_params'] if 'game_params' in params else dict()
    # my_params = params['game_params']
    # TODO nb tries
    # TODO lat-lon approx
    regatta = OffshoreRegatta(**params)
    regatta.plot(plot_weather=False)

    actions = astar(offshore_regatta=regatta)
    print(actions.shape)

    regatta.plot(plot_weather=False, track=actions)


def description():
    desc = print_description_colored_title('Astar resolution (Offshore regatta)', 'model_params')

    desc += "Game optimized by Astar"
    return desc
