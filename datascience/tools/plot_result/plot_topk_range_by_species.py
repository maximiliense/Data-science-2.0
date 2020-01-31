import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class ModelPrint(object):
    def __init__(self, name, path, color, linestyle):
        self.name = name
        self.path = path
        self.color = color
        self.linestyle = linestyle
        self.list_curves = []


def print_topk_range_plot(list_models, path="/home/bdeneu/results/"):
    grad = np.arange(100)+1

    for model in list_models:
        plt.plot(grad, np.load(path+model.path), color=model.color, linestyle=model.linestyle, label=model.name)

    majorLocator = MultipleLocator(5)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(1)

    axes = plt.gca()
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 0.40)
    axes.set_xlabel('k')
    axes.set_ylabel('mean topk accuracy by species')

    axes.xaxis.set_major_locator(majorLocator)
    axes.xaxis.set_major_formatter(majorFormatter)
    axes.xaxis.set_minor_locator(minorLocator)

    axes.yaxis.set_major_locator(MultipleLocator(0.05))
    axes.yaxis.set_minor_locator(MultipleLocator(0.01))

    plt.grid()
    plt.legend(loc=2)

    plt.show()


if __name__ == "__main__":
    list_models = []
    list_models.append(ModelPrint("CNN", "inception_rs_normal_do07_4_result_range_top100_by_species.npy", "g", "-"))
    list_models.append(ModelPrint("DNN", "inception_rs_constant_do05_result_range_top100_by_species.npy", "b", "-"))
    list_models.append(ModelPrint("Rotations", "inception_rs_rotation_do07_result_range_top100_by_species.npy", "k", "-"))
    list_models.append(ModelPrint("Permutations", "inception_rs_permutation_do07_result_range_top100_by_species.npy", "y", "-"))
    list_models.append(ModelPrint("Structure", "inception_rs_structure_do08_result_range_top100_by_species.npy", "r", "-"))
    list_models.append(ModelPrint("Mean", "inception_rs_mean_do05_result_range_top100_by_species.npy", "m", "-"))

    # list_models.append(ModelPrint("CNN-structure", "inception_structure_do06_result_range_top100_by_species.npy", "y", "-"))

    #list_models.append(ModelPrint("RF", "rf_env_d16_result_range_top100_by_species.npy", "r", "-"))
    #list_models.append(ModelPrint("BT", "bt_env_result_range_top100_by_species.npy", "k", "-"))

    #list_models.append(ModelPrint("CNN_2", "inception_rs_normal_do07_2_result_range_top100_by_species.npy", "b", "-"))
    #list_models.append(ModelPrint("CNN_3", "inception_rs_normal_do07_3_result_range_top100_by_species.npy", "r", "-"))
    #list_models.append(ModelPrint("CNN_4", "inception_rs_normal_do07_4_result_range_top100_by_species.npy", "k", "-"))
    #list_models.append(ModelPrint("CNN_5", "inception_rs_normal_do07_5_result_range_top100_by_species.npy", "y", "-"))

    print_topk_range_plot(list_models)
