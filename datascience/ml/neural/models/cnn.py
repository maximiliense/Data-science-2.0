import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, relu=True, relu_first=False, relu_last=False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.bn1 = nn.BatchNorm2d(100, eps=0.001)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 5)
        self.bn2 = nn.BatchNorm2d(150, eps=0.001)
        self.fc1 = nn.Linear(150 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = relu
        self.relu_first = relu_first
        self.relu_last = relu_last

    def forward(self, x):
        if self.relu:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 150 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.relu_first:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(self.conv2(x))
            x = x.view(-1, 150 * 5 * 5)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        elif self.relu_last:
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = x.view(-1, 150 * 5 * 5)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(F.relu(x))
        else:
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = x.view(-1, 150 * 5 * 5)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        return x
