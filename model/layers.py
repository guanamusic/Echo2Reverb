import torch

from model.base import BaseModule


class Conv1dWithInitialization(BaseModule):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


class Conv2dWithInitialization(BaseModule):
    def __init__(self, **kwargs):
        super(Conv2dWithInitialization, self).__init__()
        self.conv2d = torch.nn.Conv2d(**kwargs)
        torch.nn.init.orthogonal_(self.conv2d.weight.data, gain=1)

    def forward(self, x):
        return self.conv2d(x)


class BatchNormBlock(BaseModule):
    def __init__(self, **kwargs):
        super(BatchNormBlock, self).__init__()
        self.batch_norm = torch.nn.BatchNorm1d(**kwargs)

    def forward(self, x):
        return self.batch_norm(x)
