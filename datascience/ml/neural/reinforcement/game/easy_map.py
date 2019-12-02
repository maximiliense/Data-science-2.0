import numpy as np
import torch
from engine.games.abstract_game import AbstractGame
from engine.util.print_colors import color


class EasyMap(AbstractGame):
    def __init__(self):
        super().__init__()

        self.game = None

        self.score_ = 0
        self.current_score = 0

        self.current_dist = None

        self.start()

    def start(self):

        self.game = np.zeros((3, 100, 100))

        self.game[0, 0, 0] = 1

        for i in range(20, 100):
            self.game[1, i, i] = 1

        self.game[2, 98, 99] = 1

        self.current_score = 0

        self.current_dist = np.sqrt(2 * np.power(99, 2))

        return self.game

    def distance(self):
        where = np.argwhere(self.game[0] == 1)
        return np.sqrt(np.power(98 - where[0, 0], 2) + np.power(99 - where[0, 1], 2))

    def compute_reward(self):
        old_distance = self.current_dist
        self.current_dist = self.distance()
        return old_distance - self.current_dist

    def get_state(self):
        return torch.from_numpy(self.game).float()

    def score(self):
        return self.game.sum()

    def action(self, action):
        # 0 G, 1 D, 2 H, 3 B
        # self.current_score += 1

        where = np.argwhere(self.game[0] == 1)
        dead = False
        if action == 0 and where[0, 1] == 0:
            dead = True
        elif action == 1 and where[0, 1] == 99:
            dead = True
        elif action == 2 and where[0, 0] == 0:
            dead = True
        elif action == 3 and where[0, 0] == 99:
            dead = True
        elif action == 0:
            self.game[0, where[0, 0], where[0, 1]] = 0
            self.game[0, where[0, 0], where[0, 1] - 1] = 1
        elif action == 1:
            self.game[0, where[0, 0], where[0, 1]] = 0
            self.game[0, where[0, 0], where[0, 1] + 1] = 1
        elif action == 2:
            self.game[0, where[0, 0], where[0, 1]] = 0
            self.game[0, where[0, 0] - 1, where[0, 1]] = 1
        elif action == 3:
            self.game[0, where[0, 0], where[0, 1]] = 0
            self.game[0, where[0, 0] + 1, where[0, 1]] = 1
        where = np.argwhere(self.game[0] == 1)

        if self.game[1, where[0, 0], where[0, 1]] == 1:
            dead = True
        if self.game[2, where[0, 0], where[0, 1]] == 1:
            self.score_ = self.current_score
            self.start()
            print('He won!')
            return self.get_state(), 10, True

        if dead:
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


if __name__ == '__main__':
    game = EasyMap()
    for i in range(101):
        game.action(3)
    print(game.score_)
