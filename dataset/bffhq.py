"""
Author: Rui Hu
All rights reserved.
Modified from https://github.com/kakaoenterprise/Learning-Debiased-Disentangled/blob/master/data/util.py
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from pathlib import Path


class bFFHQDataset(Dataset):
    def __init__(self, root, split, bias_ratio, transform=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.num_classes = 2
        self.group_num = 4
        base_folder = Path(root) / "bffhq"

        self.image2pseudo = {}

        if split == 'train':
            self.align = glob(os.path.join(base_folder, bias_ratio, 'align', "*", "*"))
            self.conflict = glob(os.path.join(base_folder, bias_ratio, 'conflict', "*", "*"))
            self.data = self.align + self.conflict
        elif split == 'val':
            split = 'valid'
            self.data = glob(os.path.join(base_folder, split, "*"))
        elif split == 'test':
            self.data = glob(os.path.join(base_folder, split, "*"))

        self.target = torch.as_tensor(
            [(int(filename.split('_')[-2]), int(filename.split('_')[-1].split('.')[0])) for filename in self.data],
            dtype=torch.long
        )
        self.group_label, self.is_aligned_label = self._setup_group_and_aligned_label()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.target[index][0]
        bias = self.is_aligned_label[index]
        image = Image.open(self.data[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = torch.as_tensor([label, bias], dtype=torch.long)
        return image, target

    def _setup_group_and_aligned_label(self):
        # group_id_list = list(range(self.num_classes, self.num_bias))
        group_label = []
        is_aligned_label = []
        for (y, place) in self.target:
            if y == 0 and place == 0:
                group_label.append(0)
                is_aligned_label.append(1)
            elif y == 0 and place == 1:
                group_label.append(1)
                is_aligned_label.append(0)
            elif y == 1 and place == 0:
                group_label.append(2)
                is_aligned_label.append(0)
            else:
                group_label.append(3)
                is_aligned_label.append(1)
        group_label = torch.as_tensor(group_label, dtype=torch.long)
        is_aligned_label = torch.as_tensor(is_aligned_label, dtype=torch.long)
        return group_label, is_aligned_label
