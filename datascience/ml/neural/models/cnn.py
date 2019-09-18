import sys

import torch
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self, relu=True, relu_first=False, relu_last=False, width=100):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, width, 5)
        self.bn1 = nn.BatchNorm2d(width, eps=0.001)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(width, width + 50, 5)
        self.bn2 = nn.BatchNorm2d(width + 50, eps=0.001)
        self.fc1 = nn.Linear((width + 50) * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = relu
        self.relu_first = relu_first
        self.relu_last = relu_last
        self.width = width

    def forward(self, x):
        if self.relu:
            x = self.pool(f.relu(self.bn1(self.conv1(x))))
            x = self.pool(f.relu(self.bn2(self.conv2(x))))

            x = x.view(-1, (self.width + 50) * 5 * 5)

            x = f.relu(self.fc1(x))
            x = f.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.relu_first:
            x = self.pool(f.relu(self.conv1(x)))
            x = self.pool(self.conv2(x))
            x = x.view(-1, (self.width + 50) * 5 * 5)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        elif self.relu_last:
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = x.view(-1, (self.width + 50) * 5 * 5)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(f.relu(x))
        else:
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = x.view(-1, (self.width + 50) * 5 * 5)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        return x


def convolutional_layer(in_f, out_f, relu=True, batchnorm=True, stride=2):
    if batchnorm:
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, 5, stride=stride),
            nn.BatchNorm2d(out_f),
            # nn.Sequential()
            nn.ReLU() if relu else nn.Sequential()  # nn.Softplus()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, 5, stride=stride),
            nn.ReLU() if relu else nn.Sequential()   # nn.Softplus()
        )


def fc_layer(in_f, out_f, relu=True, bias=True):
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=bias),
        # nn.Sequential()
        nn.ReLU() if relu else nn.Sequential()  # nn.Softplus()
    )


class CustomizableCNN(nn.Module):
    def __init__(self, conv_layers=(100, 100), linear_layers=(124,), dim_out=10, im_shape=(3, 32, 32),
                 relu=True, batchnorm=True):
        super(CustomizableCNN, self).__init__()
        rep_size = im_shape[1]
        conv_size = 5
        stride = 2
        for _ in range(len(conv_layers)):
            rep_size -= conv_size - 1
            rep_size /= stride
            rep_size = int(rep_size)

        layers = [im_shape[0] if len(im_shape) == 3 else 1] + [s for s in conv_layers]

        layers = [
            convolutional_layer(in_f, out_f, relu, batchnorm) for in_f, out_f in zip(
                layers, layers[1:])
        ]

        self.conv_layers = nn.Sequential(*layers)

        layers = [conv_layers[-1] * int(rep_size)**2] + [s for s in linear_layers] + [dim_out]

        layers = [
            fc_layer(in_f, out_f, relu) for in_f, out_f in zip(
                layers, layers[1:])
        ]

        self.fc_layers = nn.Sequential(*layers)

        self.ask_layer = -1

    def __len__(self):
        return len(self.fc_layers) + len(self.conv_layers)

    def forward(self, x, layer=sys.maxsize):
        last_relu = layer == sys.maxsize
        i = 0
        while i < layer and i < len(self.conv_layers):
            seq = self.conv_layers[i]
            for s in range(len(seq)):
                if last_relu or i != layer - 1 or type(seq[s]) is not nn.ReLU:
                    x = seq[s](x)
            i += 1
        if i != layer:
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x, 2)
        i = 0
        while i + len(self.conv_layers) < layer and i < len(self.fc_layers):
            seq = self.fc_layers[i]
            for s in range(len(seq)):
                if last_relu or i + len(self.conv_layers) != layer - 1 or type(seq[s]) is not nn.ReLU:
                    x = seq[s](x)
            i += 1
        return x

