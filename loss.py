import torch
import torch.nn as nn
import math
import numpy as np


class LDAMHingeLoss(nn.Module):
    def __init__(self, C, weights, n_data):
        super(LDAMHingeLoss, self).__init__()
        self.C = C
        self.n0 = weights[0]*n_data
        self.n1 = weights[1]*n_data

    def forward(self, output, target):
        loss = 0
        for i, y in enumerate(target):
            if y == 0 :
                delta = self.C/(self.n0)**1/4
            elif y == 1 :
                delta = self.C/(self.n1)**1/4

            exp = torch.exp(output[i, y] - delta)
            print(exp)
            loss -= torch.log(exp/(exp+torch.exp(output[i, 1-y])))
        return loss