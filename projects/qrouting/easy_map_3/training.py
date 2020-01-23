from engine.util.print_colors import color, print_description_colored_title
from engine.setups.util import setup_and_run_experiment
from engine.training_validation.loss import MSELoss
from contrib.games.easy_map_3 import EasyMap
from contrib.models.cnn_tuple import Net


def run(xp_name, params, epoch=0, export=False, email=0, file=False):
    # Ce que je veux

    setup_and_run_experiment(xp_name, EasyMap, params, Net, MSELoss(), epoch=epoch, export=export,
                             email=email, file=file)


def description():
    desc = print_description_colored_title('Demo example', 'model_params')

    desc += "This model accepts 64x64x80 tensors. It is composed of 1d convolutions.\n"
    desc += "The idea is to learn the distribution of each variables independently \n"
    desc += "of the spatial structure.\n\n"
    desc += "Parameters accepted by the model:\n"
    desc += "\t- [" + color.RED + "s_conv_layers" + color.END + "]: the size of each convolutional hidden layer,\n"
    desc += "\t- [" + color.RED + "n_conv_layers" + color.END + "]: the number of hidden convolutional layers,\n"
    desc += "\t- [" + color.RED + "s_layer" + color.END + "]: the size of each fully connected hidden layer,\n"
    desc += "\t- [" + color.RED + "n_layer" + color.END + "]: the number of fully connected hidden layers."
    return desc
