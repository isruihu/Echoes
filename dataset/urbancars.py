"""
Author: Rui Hu
All rights reserved.
Modified from https://github.com/facebookresearch/Whac-A-Mole/blob/main/dataset/urbancars.py
"""

import os
import glob
import torch
import random
import numpy as np

from PIL import Image
from torch import LongTensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class UrbanCars(Dataset):
    base_folder = "urbancars"

    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    data_distribution = [
        {"obj": 0, "bg": 0, "co_obj": 0},
        {"obj": 0, "bg": 0, "co_obj": 1},
        {"obj": 0, "bg": 1, "co_obj": 0},
        {"obj": 0, "bg": 1, "co_obj": 1},
        {"obj": 1, "bg": 1, "co_obj": 1},
        {"obj": 1, "bg": 1, "co_obj": 0},
        {"obj": 1, "bg": 0, "co_obj": 1},
        {"obj": 1, "bg": 0, "co_obj": 0}
    ]

    def __init__(
        self,
        root,
        split,
        img_mode="complete",
        transform=None
    ):
        assert img_mode in ["complete", "only_target", "only_bg", "only_co_obj"]
        self.img_mode = img_mode
        self.transform = transform

        if split == "train":
            bg_ratio = 0.95
            co_occur_obj_ratio = 0.95
        elif split in ["val", "test"]:
            bg_ratio = 0.5
            co_occur_obj_ratio = 0.5
        else:
            raise NotImplementedError

        ratio_combination_folder_name = (
            f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
        )
        img_root = os.path.join(
            root, self.base_folder, ratio_combination_folder_name, split
        )

        self.img_fpath_list = []
        self.obj_bg_co_occur_obj_label_list = []

        for obj_id, obj_name in enumerate(self.obj_name_list):
            for bg_id, bg_name in enumerate(self.bg_name_list):
                for co_occur_obj_id, co_occur_obj_name in enumerate(
                    self.co_occur_obj_name_list
                ):
                    dir_name = (
                        f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                    )
                    dir_path = os.path.join(img_root, dir_name)
                    assert os.path.exists(dir_path)

                    img_fpath_list = glob.glob(os.path.join(dir_path, "*.jpg"))
                    self.img_fpath_list += img_fpath_list
                    self.obj_bg_co_occur_obj_label_list += [
                        (obj_id, bg_id, co_occur_obj_id)
                    ] * len(img_fpath_list)

        if img_mode == "complete":
            pass
        elif img_mode == "only_target":
            self.img_fpath_list = [img_fpath.replace(".jpg", "_only_target.png") for img_fpath in self.img_fpath_list]
        elif img_mode == "only_bg":
            self.img_fpath_list = [img_fpath.replace(".jpg", "_only_background.png") for img_fpath in self.img_fpath_list]
        elif img_mode == "only_co_obj":
            self.img_fpath_list = [img_fpath.replace(".jpg", "_only_co_obj.png") for img_fpath in self.img_fpath_list]
        else:
            raise NotImplementedError

        self.target = torch.as_tensor(self.obj_bg_co_occur_obj_label_list, dtype=torch.long)
        self.group_num = 8
        self.group_label, self.biasA_is_aligned, self.biasB_is_aligned = self._setup_group_label_is_aligned()

    def __len__(self):
        return len(self.img_fpath_list)

    def __getitem__(self, index):
        img_fpath = self.img_fpath_list[index]

        if self.img_mode in ["complete", "only_target"]:
            target = self.obj_bg_co_occur_obj_label_list[index]
        elif self.img_mode == "only_bg":
            target = [self.obj_bg_co_occur_obj_label_list[index][1]]
        elif self.img_mode == "only_co_obj":
            target = [self.obj_bg_co_occur_obj_label_list[index][2]]
        else:
            raise NotImplementedError

        img = Image.open(img_fpath)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        target = torch.LongTensor(target)
        return img, target

    def _setup_group_label_is_aligned(self):
        group_list = self.data_distribution
        group_label = []
        for _target in self.target:
            for group_id, group_info in enumerate(group_list):
                if (
                        (_target[0] == group_info['obj']) &
                        (_target[1] == group_info['bg']) &
                        (_target[2] == group_info['co_obj'])
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


def get_transform(split):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transform

