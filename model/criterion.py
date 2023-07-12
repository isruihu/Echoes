"""
Author: Rui Hu
All rights reserved.
"""

import torch
import torch.nn.functional as F
import numpy as np


class GeneralizedCELoss(torch.nn.Module):
    def __init__(self, q=0.7, reduction='mean'):
        super(GeneralizedCELoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.q = q
        self.reduction = reduction

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
