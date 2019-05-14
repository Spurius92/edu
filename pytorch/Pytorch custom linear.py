#utf-8
# custom class for nn.Linear in PyTorch

import torch
import torch.nn as nn
import torch.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return F.lin(input, self.weight, self.bias)
        