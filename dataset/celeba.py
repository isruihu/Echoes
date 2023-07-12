"""
Author: Rui Hu
All rights reserved.
"""

import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.datasets as tv_dataset


class BiasedCelebA(Dataset):
    raw_celeba = tv_dataset.CelebA('data/', 'all')

    def __init__(self,
                 root,
                 target_name,
                 biasA_name,
                 biasB_name,
                 biasA_ratio,
                 biasB_ratio,
                 split,
                 transform=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform

        base_folder = f"celeba-MB/{target_name}-{biasA_name}-{biasA_ratio}-{biasB_name}-{biasB_ratio}"
        data_dir = Path(root) / base_folder / f"{split}_data.npy"

        self.data = np.load(open(data_dir, 'rb'))
        self.target = self.data[:, 1:]

        self.group_num = 8
        self.meta_key = f"{target_name}__{biasA_name}__{biasB_name}"
        self.group_label, self.biasA_is_aligned, self.biasB_is_aligned = self._setup_group_label_is_aligned()

    def __getitem__(self, index):
        raw_index = self.data[index][0]
        target = self.data[index][1:]
        image = self.raw_celeba[raw_index][0]

        if self.transform:
            image = self.transform(image)

        target = torch.as_tensor(target)
        return image, target

    def __len__(self):
        return len(self.data)

    def _setup_group_label_is_aligned(self):
        metadata = json.load(open("create_dataset/celeba/metadata.json", 'r'))
        assert self.meta_key in metadata.keys()

        group_def = list(metadata[self.meta_key].values())
        group_list = [group_def[0][0], group_def[1][0], group_def[2][0], group_def[3][0]] + \
                    [group_def[0][1], group_def[1][1], group_def[2][1], group_def[3][1]]

        group_label = []
        for _target in self.target:
            for group_id, group_info in enumerate(group_list):
                if (
                    (_target[0] == group_info['target']) &
                    (_target[1] == group_info['biasA']) &
                    (_target[2] == group_info['biasB'])
                ):
                    group_label.append(group_id)
                    break
        group_label = torch.as_tensor(
            group_label,
            dtype=torch.long
        )

        biasA_is_aligned = torch.as_tensor(
            (group_label == 0) | (group_label == 1) | (group_label == 4) | (group_label == 5),
            dtype=torch.long
        )
        biasB_is_aligned = torch.as_tensor(
            (group_label == 0) | (group_label == 2) | (group_label == 4) | (group_label == 6),
            dtype=torch.long
        )

        return group_label, biasA_is_aligned, biasB_is_aligned
