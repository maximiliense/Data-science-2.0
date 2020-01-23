from abc import ABC, abstractmethod
from torch import nn


class Loss(ABC):
    """
    Abstract class for loss definition
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, output, label):
        pass

    def __call__(self, output, label):
        return self.loss(output, label)


class CELoss(Loss):
    """
    Cross entropy loss
    """

    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

    def loss(self, output, label):
        return self.criterion(output, label)


class MSELoss(Loss):
    """
    Mean Square Error loss
    """

    def __init__(self):
        super().__init__()

        self.criterion = nn.MSELoss()

    def loss(self, output, label):
        return self.criterion(output, label)
