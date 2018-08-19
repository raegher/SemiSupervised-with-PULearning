import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import torch.nn.init as nn_init
from torch import tensor


class nnPUloss(nn.Module):
    def __init__(self, prior, alpha=1., beta=0., gamma=1.):
        super(nnPUloss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.unlabled = -1
        self.CEL = nn.CrossEntropyLoss(reduce=False)
        self.prior = prior

    def forward(self, *input):
        x, y = input
        unlabled = (y == self.unlabled)
        labled = ~unlabled
        n_unlabled = torch.tensor(max([1.0, torch.sum(unlabled)]), dtype=torch.float32)
        n_cls = len(self.prior)
        self.loss = None
        unrisk = []
        inds_y = []
        n_inds_y = []
        for i in range(n_cls):
            inds_y.append(torch.tensor((y == i) * labled, dtype=torch.float32))
            n_inds_y.append(torch.tensor(max([1., torch.sum(inds_y[i])]), dtype=torch.float32))
        for i in range(n_cls):
            tmpl_i = torch.ones(y.shape, dtype=torch.int32)*i
            losses = torch.tensor(self.CEL(x, tmpl_i), dtype=torch.float32)
            unrisk.append(torch.sum(losses * torch.tensor(unlabled, dtype=torch.float32)) / n_unlabled)
            nn_risk_y = 0
            for j in range(n_cls):
                if j != i:
                    nn_risk_y += torch.sum(losses * inds_y[j]) / n_inds_y[j] * self.prior[j]

            self.loss = torch.sum(losses * inds_y[i]) / n_inds_y[i] * self.prior[i]
            self.loss += self.alpha * max([unrisk[i] - nn_risk_y, self.beta]) / n_cls

        return self.loss
