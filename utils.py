"""
Author: Rui Hu
All rights reserved.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn


EPS = 1e-6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, bit=6):
        self.reset()
        self.bit = bit

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, correct_num, batch_size):
        self.val = correct_num
        self.sum += correct_num
        self.count += batch_size
        self.avg = self.sum / self.count


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx, self.dataset[idx]


class EMAGPU:
    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0)).cuda()
        self.updated = torch.zeros(label.size(0)).cuda()
        self.num_class = label.max().item() + 1
        self.max_param_per_class = torch.zeros(self.num_class).cuda()

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index]
            + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

        # update max_param_per_class
        batch_size = len(index)
        buffer = torch.zeros(batch_size, self.num_class).cuda()
        buffer[range(batch_size), self.label[index]] = self.parameter[index]
        cur_max = buffer.max(dim=0).values
        global_max = torch.maximum(cur_max, self.max_param_per_class)
        label_set_indices = self.label[index].unique()
        self.max_param_per_class[label_set_indices] = global_max[
            label_set_indices
        ]

    def max_loss(self, label):
        return self.max_param_per_class[label]