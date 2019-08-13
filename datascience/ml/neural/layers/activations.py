import torch.nn.functional as F
from torch.nn.modules.module import Module


class StickyReLU(Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(StickyReLU, self).__init__()
        self.inplace = inplace
        self.x_dropout = {}

    def forward(self, _input):
        return F.relu(_input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str
