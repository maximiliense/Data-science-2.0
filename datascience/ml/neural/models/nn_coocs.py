import torch.nn as nn
import torch.nn.functional as F


class NNCoocs(nn.Module):

    def __init__(self, n_labels=3336, last_layer=False):
        super(NNCoocs, self).__init__()

        # co-occurrences
        self.fc_cooc_1 = nn.Linear(n_labels, 256)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, n_labels)

        self.last_layer = last_layer

    def forward(self, x):

        # n_labels
        x = self.fc_cooc_1(x)
        x = self.bn(x)
        x = self.relu(x)

        if not self.last_layer:
            # 256
            x = self.fc(x)
            # (num_classes)
            if not self.training:
                x = F.softmax(x, dim=-1)

        return x
