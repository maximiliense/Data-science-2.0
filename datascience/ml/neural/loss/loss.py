import json
import os

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from datascience.visu.util.util import plt, save_fig_direct_call
from engine.hardware import use_gpu
from engine.logging import print_info
from engine.path import output_path


class Loss(ABC):
    """
    Abstract class for loss definition
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, output, label):
        pass

    @staticmethod
    def output(output):
        return output

    def __call__(self, output, label):
        return self.loss(output, label)

    def __repr__(self):
        return 'Loss'


class CEMTLossSpeciesArea(Loss):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.configured = False
        self.criterion_species = None
        self.criterion_area = None
        self.r = 0.75

    def configure(self):
        # this is here because it needs to be done after GPU configuration...
        weight = torch.FloatTensor([0 if i == 0 else 1 for i in range(self.dim)])
        weight = weight.cuda()
        self.criterion_species = nn.CrossEntropyLoss(weight=weight)
        self.criterion_area = nn.CrossEntropyLoss()

    def loss(self, output, label):
        if not self.configured:
            self.configure()

        return self.r*self.criterion_species(output[0], label[0]) + (1-self.r)*self.criterion_area(output[1], label[1])

    @staticmethod
    def output(output):
        return output[0]

    def __repr__(self):
        return 'Multi-task Cross Entropy (Species/Area)'


class CELoss(Loss):
    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def loss(self, output, label):
        return self.criterion(output, label)

    def __repr__(self):
        return 'Cross Entropy'


class CategoricalPoissonLoss(Loss):
    def __init__(self, log_input=True):
        super().__init__()
        self.criterion = nn.PoissonNLLLoss(log_input=log_input)

    def loss(self, output, label):
        # constructing a one hot encoding structure
        one_hot = torch.zeros(size=output.size())

        # eventually put it on the GPU
        if use_gpu():
            one_hot = one_hot.cuda()

        # set the same number of dimension to label
        label = label.long().unsqueeze(1)

        # fill one_hot based on the label
        one_hot = one_hot.scatter_(1, label, 1.)
        return self.criterion(output.view(-1), one_hot.view(-1))

    def __repr__(self):
        return 'Categorical Poisson loss'


class CELossBayesian(Loss):
    def __init__(self, prior):
        super().__init__()
        # Variable(torch.from_numpy(np.log(self.prior)).float())

        # eventually set the prior on the GPU
        if use_gpu():
            self.prior = torch.from_numpy(np.log(prior)).float().cuda()
        else:
            self.prior = torch.from_numpy(np.log(prior)).float()

        self.criterion = nn.CrossEntropyLoss()

    def loss(self, output, label):
        return self.criterion(output + self.prior, label)

    def __repr__(self):
        return 'Bayesian Cross Entropy'


def save_loss(losses, ylabel='Loss'):

    min_freq = min(losses.values(), key=lambda x: x[1])[1]
    if min_freq == 0:
        return
    plt('loss').title('Losses curve')
    plt('loss').xlabel('x' + str(min_freq) + ' batches')
    plt('loss').ylabel(ylabel)

    for k in losses:

        offset = losses[k][1] // min_freq - 1
        plt('loss').plot(
            # in order to align the multiple losses
            [i for i in range(offset, len(losses[k][0]) * (losses[k][1] // min_freq) + offset, losses[k][1] // min_freq)],
            losses[k][0], label=k
        )

        _json = json.dumps(losses[k][0])
        path = output_path('loss_{}.logs'.format(k))
        f = open(path, "w")
        f.write(_json)
        f.close()
    plt('loss').legend()
    save_fig_direct_call(figure_name='loss')


def load_loss(name):
    path = output_path(name + '.logs')
    print_info('Loading loss at ' + path)
    if os.path.exists(path):
        with open(path) as f:
            loss = json.load(f)
        return loss
    return []
