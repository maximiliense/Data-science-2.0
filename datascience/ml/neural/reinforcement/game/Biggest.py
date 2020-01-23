import numpy as np
import torch
from engine.games.abstract_game import AbstractGame


class Biggest(AbstractGame):
    def __init__(self):
        super().__init__()

        self.game = None
        self.nb_actions = 3

        self.score_ = 0

        self.start()

    def start(self):
        self.nb_actions = 3
        self.game = np.random.randint(0, 6, size=(5,))
        return self.game

    def get_state(self):

        state = torch.from_numpy(np.concatenate(([self.nb_actions], self.game))).float()

        return state

    def score(self):
        return self.game.sum()

    def action(self, action):
        if action >= len(self.game):
            gain = 0
        else:
            current_score = self.score()
            self.game[action] = np.random.randint(0, 6)
            gain = self.score() - current_score

        self.nb_actions -= 1
        if self.nb_actions == 0:
            self.score_ = sum(self.game)
            self.start()
            return self.get_state(), gain, True
        else:
            return self.get_state(), gain, False

    def print(self):
        print(str(self.game.sum()) + " : " + str(self.game) + " : " + str(self.nb_actions))


if __name__ == '__main__':
    game = Biggest()
    print(game.score())
    print(game.game)
