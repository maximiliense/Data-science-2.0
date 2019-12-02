import sys

import torch.nn as nn
import ast
import torch.nn.functional as F


def fc_layer(in_f, out_f, relu=True, batchnorm=True, bias=True):
    if batchnorm:
        return nn.Sequential(
            nn.Linear(in_f, out_f, bias=bias),
            nn.BatchNorm1d(out_f),
            # nn.Sequential()
            nn.ReLU() if relu else nn.Sequential()  # nn.Softplus()
        )
    else:
        return nn.Sequential(
            nn.Linear(in_f, out_f, bias=bias),
            nn.ReLU() if relu else nn.Sequential()   # nn.Softplus()
        )


class FullyConnected(nn.Module):
    def __init__(self, n_labels=2, n_input=2, architecture=(2,), relu=True, batchnorm=True, bias=True, dropout=0.0,
                 last_sigmoid=True, hidden_layer=False):

        super(FullyConnected, self).__init__()
        self.last_sigmoid = last_sigmoid

        self.n_input = n_input
        self.n_labels = n_labels
        self.dropout = dropout
        self.hidden_layer = hidden_layer

        if type(architecture) is str:
            architecture = ast.literal_eval(architecture)
        layer_size = [n_input] + [i for i in architecture]

        layers = [
            fc_layer(in_f, out_f, relu, batchnorm, bias) for in_f, out_f in zip(layer_size, layer_size[1:])
        ]
        self.layers = nn.Sequential(*layers)

        self.linear_layer = nn.Linear(layer_size[-1], n_labels)

    def forward(self, x):
        x = self.layers(x)
        if not self.hidden_layer:
            x = self.linear_layer(x)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.last_sigmoid and not self.training:
                x = F.softmax(x, dim=-1)

        return x


class FullyConnectedDeepAnalysis(nn.Module):
    def __init__(self, n_labels=2, n_input=2, architecture=(2,), relu=True, batchnorm=True, bias=True, dropout=0.0,
                 last_sigmoid=True, hidden_layer=False):
        super(FullyConnectedDeepAnalysis, self).__init__()
        if type(architecture) is str:
            architecture = ast.literal_eval(architecture)
        self.last_sigmoid = last_sigmoid
        layer_size = [n_input] + [i for i in architecture] + [n_labels]
        self.length = len(architecture) + 1

        layers = [
            fc_layer(in_f, out_f, relu, batchnorm, bias) for in_f, out_f in zip(layer_size, layer_size[1:])
        ]
        self.layers = nn.Sequential(*layers)

        self.n_input = n_input
        self.n_labels = n_labels
        self.dropout = dropout
        self.hidden_layer = hidden_layer

    def __len__(self):
        return self.length

    def forward(self, x, layer=sys.maxsize):
        last_relu = layer == sys.maxsize
        i = 0

        while i < layer and i < len(self):
            seq = self.layers[i]
            for s in range(len(seq)):
                if last_relu or i != layer - 1 or type(seq[s]) is not nn.ReLU:
                    x = seq[s](x)
            i += 1

        if self.last_sigmoid and not self.training:
            x = F.softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    model = FullyConnectedDeepAnalysis()
    print(model)
