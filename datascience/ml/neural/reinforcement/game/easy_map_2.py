import numpy as np
import torch
from engine.games.abstract_game import AbstractGame
import matplotlib.pyplot as plt

from engine.util.print_colors import color


class EasyMap(AbstractGame):
    def __init__(self):
        super().__init__()

        self.game = None

        self.position = [100, 100]

        self.score_ = 0
        self.current_score = 0

        self.current_dist = None
        self.plot_mpl = None

        self.start()
        self.plot()

    def start(self):

        self.game = np.zeros((2, 300, 300))
        self.game[0, 0:100, :] = 1
        self.game[0, 200:300, :] = 1
        self.game[0, 100:200, :100] = 1
        self.game[0, 100:200, 200:300] = 1

        for i in range(120, 200):
            self.game[0, 300 - i, i] = 1
            self.game[0, 301 - i, i] = 1
            self.game[0, 299 - i, i] = 1

        self.game[1, 199, 199] = 1

        self.current_score = 0
        self.position = np.random.randint(100, 200, size=2)

        while self.game[0, self.position[0], self.position[1]] == 1:
            self.position = np.random.randint(100, 200, size=2)
        self.current_dist = self.distance()

        return self.game

    def distance(self):
        return np.sqrt(np.power(198 - self.position[0], 2) + np.power(199 - self.position[1], 2))

    def compute_reward(self):
        old_distance = self.current_dist
        self.current_dist = self.distance()
        return old_distance - self.current_dist

    def get_state(self):
        return torch.from_numpy(self.game[:, self.position[0] - 50:self.position[0] + 50,
                                self.position[1] - 50:self.position[1] + 50]).float()

    def score(self):
        return self.game.sum()

    def action(self, action):
        # 0 G, 1 D, 2 H, 3 B
        # self.current_score += 1

        dead = False
        if action == 0 and self.position[1] == 0:
            dead = True
        elif action == 1 and self.position[1] == 99:
            dead = True
        elif action == 2 and self.position[0] == 0:
            dead = True
        elif action == 3 and self.position[0] == 99:
            dead = True
        elif action == 0:
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
            self.start()
            print(color.GREEN + 'He won!' + color.END)
            return self.get_state(), 10, True
        # print(min(max(self.position[0] - 100, 0), 99))
        # print(min(max(self.position[1] - 100, 0), 99))
        self.plot_mpl[min(max(self.position[0] - 100, 0), 99), min(max(self.position[1] - 100, 0), 99)] = 5
        if dead:
            print(color.RED + 'he is dead' + color.END)
            self.score_ = self.current_score
            self.start()
            return self.get_state(), -10, True
        else:
            rew = self.compute_reward()
            self.current_score += rew
            # print(str(where) + ' : ' + str(self.compute_reward()))
            return self.get_state(), rew, False

    def print(self):
        print(str(self.game))

    def plot(self):

        self.plot_mpl = np.copy(self.game[0][100:200, 100:200])
        self.plot_mpl[self.position[0] - 100, self.position[1] - 100] = 5
        self.plot_mpl[self.game[1][100:200, 100:200] == 1] = 3
        # print(self.plot_mpl.shape)

    def save_plot(self, path=None):
        plt.figure()

        plt.imshow(self.plot_mpl)
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()


if __name__ == '__main__':
    game = EasyMap()
    for _ in range(50):
        game.action(1)

    for _ in range(49):
        game.action(3)
    for _ in range(49):
        game.action(1)
    for _ in range(49):
        game.action(3)

    # print(game.get_state().shape)
    game.save_plot()
    # print(game.action(3))
