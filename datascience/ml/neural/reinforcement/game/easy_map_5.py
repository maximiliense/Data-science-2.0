import numpy as np
import torch
from engine.games.abstract_game import AbstractGame
import matplotlib.pyplot as plt

from engine.util.print_colors import color


# with timelag
class EasyMap(AbstractGame):
    def __init__(self, nb_obstacles=30, max_height=500, max_width=500, nb_coup_max=1000, lag_size=5):
        super().__init__()

        self.game = None

        self.position = None

        self.target = None

        self.score_ = 0
        self.current_score = 0
        self.nb_coup = 0

        self.nb_coup_max = nb_coup_max

        self.current_dist = None
        self.plot_mpl = None

        self.width = None
        self.height = None

        self.nb_obstacles = nb_obstacles

        self.max_height = max_height
        self.max_width = max_width

        self.old_plot_mpl = None

        self.lag = None
        self.lag_size = lag_size

        self.start()

    def start(self):
        # size window
        self.width = 110 + np.random.randint(100, self.max_width)
        self.height = 110 + np.random.randint(100, self.max_height)

        # constructing board
        self.game = np.zeros((2, self.height, self.width))

        self.game[0, 0:55, :] = 1
        self.game[0, self.height - 55:self.height, :] = 1

        self.game[0, :, 0:55] = 1
        self.game[0, :, self.width - 55:self.width] = 1

        # position target
        self.target = [np.random.randint(56, self.height - 56), np.random.randint(56, self.width - 56)]
        self.game[1, self.target[0] - 2:self.target[0] + 2, self.target[1] - 2:self.target[1] + 2] = 1

        # position player:
        self.position = [np.random.randint(56, self.height - 56), np.random.randint(56, self.width - 56)]
        p0 = self.position[0]
        p1 = self.position[1]
        while self.game[0, p0, p1] == 1 or self.game[0, p0, p1] == 1:
            self.position = [np.random.randint(56, self.height - 56), np.random.randint(56, self.width - 56)]

        self.game[1, self.position[0], self.position[1]] = 1

        nb_obstacles = np.random.randint(5, self.nb_obstacles)
        obstacles_done = 0
        while obstacles_done < nb_obstacles:
            center = [np.random.randint(56, self.height - 56), np.random.randint(56, self.width - 56)]

            if len(np.argwhere(self.game[1, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16])) == 0:
                self.game[0, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = 1
                obstacles_done += 1

        self.game[1, self.position[0], self.position[1]] = 0

        self.current_dist = self.distance()
        self.current_score = 0
        self.nb_coup = 0

        self.plot()

        self.lag = np.concatenate([self.get_view() for _ in range(self.lag_size)])

        return self.game

    def direction(self):
        return np.array([self.position[0] - self.target[0], self.position[1] - self.target[1]])

    def distance(self):
        return np.sqrt(np.power(self.position[0] - self.target[0], 2) + np.power(self.position[1] - self.target[1], 2))

    def compute_reward(self):
        old_distance = self.current_dist
        self.current_dist = self.distance()
        return old_distance - self.current_dist

    def get_view(self):
        return self.game[:, self.position[0] - 25:self.position[0] + 25,
               self.position[1] - 25:self.position[1] + 25]

    def get_state(self):
        self.lag = np.concatenate([self.lag[2:], self.get_view()])

        return torch.from_numpy(self.lag).float(), torch.from_numpy(
            self.direction()).float()

    def score(self):
        return self.game.sum()

    def action(self, action):
        self.nb_coup += 1
        # 0 G, 1 D, 2 H, 3 B
        # self.current_score += 1

        dead = False

        if action == 0:
            self.position[1] -= 1
        elif action == 1:
            self.position[1] += 1
        elif action == 2:
            self.position[0] -= 1
        elif action == 3:
            self.position[0] += 1

        if self.game[0, self.position[0], self.position[1]] == 1:
            dead = True

        if self.game[1, self.position[0], self.position[1]] == 1:
            self.score_ = self.current_score
            self.old_plot_mpl = np.copy(self.plot_mpl)
            self.start()
            print(color.GREEN + 'He won!' + color.END)
            return self.get_state(), 10, True
        # print(min(max(self.position[0] - 100, 0), 99))
        # print(min(max(self.position[1] - 100, 0), 99))

        self.plot_mpl[self.position[0], self.position[1]] = 5
        if dead:
            print(color.RED + 'he is dead' + color.END)
            self.score_ = self.current_score
            self.old_plot_mpl = np.copy(self.plot_mpl)
            self.start()
            return self.get_state(), -10, True
        elif self.nb_coup >= self.nb_coup_max:
            print(color.YELLOW + 'restarting' + color.END)
            self.score_ = self.current_score
            self.old_plot_mpl = np.copy(self.plot_mpl)
            self.start()
            return self.get_state(), -1, True
        else:
            rew = self.compute_reward()
            self.current_score += rew
            # print(str(where) + ' : ' + str(self.compute_reward()))
            return self.get_state(), rew, False

    def print(self):
        print(str(self.game))

    def plot(self):

        self.plot_mpl = np.copy(self.game[0])
        self.plot_mpl[self.position[0] - 2:self.position[0] + 2,
        self.position[1] - 2:self.position[1] + 2] = 5
        self.plot_mpl[self.game[1] == 1] = 3
        # print(self.plot_mpl.shape)

    def save_plot(self, path=None, old=False):
        plt.figure()
        pl = self.old_plot_mpl if old and self.old_plot_mpl is not None else self.plot_mpl
        plt.imshow(pl)
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    def show_view(self):
        plt.figure()

        plot_mpl = np.copy(self.game[0, self.position[0] - 25:self.position[0] + 25,
                           self.position[1] - 25:self.position[1] + 25])
        plt_mpl2 = np.copy(self.game[1, self.position[0] - 25:self.position[0] + 25,
                           self.position[1] - 25:self.position[1] + 25])

        plot_mpl[plt_mpl2 == 1] = 3

        plt.imshow(plot_mpl)
        plt.show()
        plt.close()


if __name__ == '__main__':
    game = EasyMap()
    game.save_plot()
    msg = input()
    while msg != 'exit':
        print(game.action(int(msg))[1])
        print(game.get_state()[1])
        game.save_plot()
        msg = input()
    exit()
    for _ in range(25):
        print(game.action(1)[1])
    game.save_plot(old=True)
    exit()

    for _ in range(25):
        game.action(3)
    for _ in range(25):
        game.action(1)
    for _ in range(25):
        game.action(3)

    # print(game.get_state().shape)
    game.save_plot()
    game.show_view()
    print(game.direction())
    # print(game.action(3))
