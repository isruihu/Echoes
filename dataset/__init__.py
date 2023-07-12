"""
Author: Rui Hu
All rights reserved.
"""

import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .celeba import BiasedCelebA
from .urbancars import UrbanCars
from .bffhq import bFFHQDataset


class Subset(Dataset):
    def __init__(self, dataset, limit):
        self.dataset = [dataset[i] for i in range(limit)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def get_dataset(dataset_name, split, args):
    if 'pct' in dataset_name:
        transform = get_transform(dataset_name.rsplit('_', 1)[0], split)
    else:
        transform = get_transform(dataset_name, split)

    if dataset_name == 'celeba':
        idx2attr = json.load(
            open("create_dataset/celeba/idx2attr.json", 'r')
        )
        idx2attr = {int(k): v for k, v in idx2attr.items()}
        target_name = idx2attr[args.target_id]
        biasA_name = idx2attr[args.biasA_id]
        biasB_name = idx2attr[args.biasB_id]
        # celeba dataset
        dataset = BiasedCelebA(
            root=args.root,
            target_name=target_name,
            biasA_name=biasA_name,
            biasB_name=biasB_name,
            biasA_ratio=args.biasA_ratio,
            biasB_ratio=args.biasB_ratio,
            split=split,
            transform=transform
        )
    elif dataset_name == 'urbancars':
        dataset = UrbanCars(
            root=args.root,
            split=split,
            img_mode="complete",
            transform=transform
        )
    elif 'bffhq' in dataset_name:
        bias_ratio = dataset_name.split('_')[-1]
        dataset = bFFHQDataset(
            root=args.root,
            split=split,
            bias_ratio=bias_ratio,
            transform=transform
        )
    else:
        raise NotImplementedError

    return dataset


def get_transform(dataset_name, split):
    train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    dataset2transform = {
        'celeba': {
            'train': train_transform,
            'val': val_transform,
            'test': val_transform
        },
        'urbancars': {
            'train': train_transform,
            'val': val_transform,
            'test': val_transform
        },
        'bffhq': {
            'train': train_transform,
            'val': val_transform,
            'test': val_transform
        },
    }

    return dataset2transform[dataset_name][split]
