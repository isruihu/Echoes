"""
Author: Rui Hu
All rights reserved.
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torchvision.datasets as tv_dataset


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='data/')
    parser.add_argument("--gen_data_dir", type=str, default="data/celeba-MB")
    parser.add_argument("--save_imgs", action="store_true")
    parser.add_argument("--num_train", type=int, default=4000)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--target_id", type=int)
    parser.add_argument("--biasA_id", type=int)
    parser.add_argument("--biasB_id", type=int)
    parser.add_argument("--biasA_ratio", type=float, default=0.95)
    parser.add_argument("--biasB_ratio", type=float, default=0.95)

    parser.add_argument("--view", action="store_true")

    args = parser.parse_args()
    print(args)
    return args


class CelebAGEN:
    def __init__(self, args):
        self.args = args
        self.raw_celeba = tv_dataset.CelebA(args.root, 'all')
        self.attr_matrix = self.raw_celeba.attr.numpy()
        idx2attr = json.load(
            open(Path(os.path.dirname(__file__)) / "idx2attr.json", 'r')
        )
        self.idx2attr = {int(k): v for k, v in idx2attr.items()}
        self.data_distribution = json.load(
            open(Path(os.path.dirname(__file__)) / "metadata.json", 'r')
        )

    def _get_biased_indices(self):
        args = self.args
        target_id = args.target_id
        biasA_id = args.biasA_id
        biasB_id = args.biasB_id

        key = "__".join([self.idx2attr[target_id],
                         self.idx2attr[biasA_id],
                         self.idx2attr[biasB_id]])

        assert key in self.data_distribution.keys()
        data_dis = self.data_distribution[key]

        target_vector = self.attr_matrix[:, target_id]
        biasA_vector = self.attr_matrix[:, biasA_id]
        biasB_vector = self.attr_matrix[:, biasB_id]

        train_indices = []
        val_indices = []
        test_indices = []

        for key, value in data_dis.items():
            for dic in value:
                t, bA, bB = dic['target'], dic['biasA'], dic['biasB']
                group_indices = np.where((target_vector == t) &
                                         (biasA_vector == bA) &
                                         (biasB_vector == bB)
                                         )[0]

                if key == "common_biasA_common_biasB":
                    group_train_num = args.num_train * args.biasA_ratio * args.biasB_ratio
                elif key == "common_biasA_uncommon_biasB":
                    group_train_num = args.num_train * args.biasA_ratio * (1 - args.biasB_ratio)
                elif key == "uncommon_biasA_common_biasB":
                    group_train_num = args.num_train * (1 - args.biasA_ratio) * args.biasB_ratio
                elif key == "uncommon_biasA_uncommon_biasB":
                    group_train_num = args.num_train * (1 - args.biasA_ratio) * (1 - args.biasB_ratio)
                else:
                    raise KeyError
                group_train_num = int(group_train_num)
                group_val_num = args.num_val * 0.25
                group_test_num = args.num_test * 0.25
                split_num1 = int(group_train_num + group_val_num)
                split_num2 = int(group_train_num + group_val_num + group_test_num)

                train_select = group_indices[:group_train_num]
                val_select = group_indices[group_train_num: split_num1]
                test_select = group_indices[split_num1: split_num2]

                train_indices.append(train_select)
                val_indices.append(val_select)
                test_indices.append(test_select)

        train_indices = np.concatenate(train_indices, axis=0)
        val_indices = np.concatenate(val_indices, axis=0)
        test_indices = np.concatenate(test_indices, axis=0)
        return train_indices, val_indices, test_indices

    def binary_attribute_count(self, target_id, attr1_id, attr2_id, attr_matrix=None):
        idx2attr = self.idx2attr
        if attr_matrix is None:
            attr_matrix = self.attr_matrix

        target_vector = attr_matrix[:, target_id]
        attr1_vector = attr_matrix[:, attr1_id]
        attr2_vector = attr_matrix[:, attr2_id]

        res = [[0, 0, 0, 0], [0, 0, 0, 0]]

        res[0][0] = np.sum((target_vector == 0) & (attr1_vector == 0) & (attr2_vector == 0))
        res[0][1] = np.sum((target_vector == 0) & (attr1_vector == 0) & (attr2_vector == 1))
        res[0][2] = np.sum((target_vector == 0) & (attr1_vector == 1) & (attr2_vector == 0))
        res[0][3] = np.sum((target_vector == 0) & (attr1_vector == 1) & (attr2_vector == 1))

        res[1][0] = np.sum((target_vector == 1) & (attr1_vector == 0) & (attr2_vector == 0))
        res[1][1] = np.sum((target_vector == 1) & (attr1_vector == 0) & (attr2_vector == 1))
        res[1][2] = np.sum((target_vector == 1) & (attr1_vector == 1) & (attr2_vector == 0))
        res[1][3] = np.sum((target_vector == 1) & (attr1_vector == 1) & (attr2_vector == 1))

        data = [
            [f"not {idx2attr[target_id]}", res[0][0], res[0][1], res[0][2], res[0][3]],
            [f"{idx2attr[target_id]}", res[1][0], res[1][1], res[1][2], res[1][3]]
        ]
        df = pd.DataFrame(data)
        df.columns = ['',
                      f'{idx2attr[attr1_id]}:0 {idx2attr[attr2_id]}:0',
                      f'{idx2attr[attr1_id]}:0 {idx2attr[attr2_id]}:1',
                      f'{idx2attr[attr1_id]}:1 {idx2attr[attr2_id]}:0',
                      f'{idx2attr[attr1_id]}:1 {idx2attr[attr2_id]}:1'
                      ]
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 500)
        print(df)
        sum = np.sum(res)
        print(f'count {sum}\n')

    def gen_data(self):
        args = self.args

        train_indices, val_indices, test_indices = self._get_biased_indices()
        os.makedirs(args.gen_data_dir, exist_ok=True)

        target = self.idx2attr[args.target_id]
        biasA = self.idx2attr[args.biasA_id]
        biasB = self.idx2attr[args.biasB_id]

        print(f"Target: {target} Bias: {biasA} & {biasB}")

        data_dir = Path(args.gen_data_dir) / f"{target}-{biasA}-{args.biasA_ratio}-{biasB}-{args.biasB_ratio}"
        os.makedirs(data_dir, exist_ok=True)

        train_data = np.stack([
            train_indices,
            self.attr_matrix[train_indices][:, args.target_id],
            self.attr_matrix[train_indices][:, args.biasA_id],
            self.attr_matrix[train_indices][:, args.biasB_id]
        ], axis=1)
        val_data = np.stack([
            val_indices,
            self.attr_matrix[val_indices][:, args.target_id],
            self.attr_matrix[val_indices][:, args.biasA_id],
            self.attr_matrix[val_indices][:, args.biasB_id]
        ], axis=1)
        test_data = np.stack([
            test_indices,
            self.attr_matrix[test_indices][:, args.target_id],
            self.attr_matrix[test_indices][:, args.biasA_id],
            self.attr_matrix[test_indices][:, args.biasB_id]
        ], axis=1)
        np.save(open(data_dir / "train_data.npy", 'wb'), train_data)
        np.save(open(data_dir / "val_data.npy", 'wb'), val_data)
        np.save(open(data_dir / "test_data.npy", 'wb'), test_data)

        if args.save_imgs:
            for split, data in zip(
                ["train", "val", "test"],
                [train_data, val_data, test_data]
            ):
                os.makedirs(data_dir / f"images/{split}", exist_ok=False)
                for iii, (idx, target, biasA, biasB) in enumerate(tqdm(data)):
                    img = self.raw_celeba[idx][0]
                    fpath = data_dir / f'images/{split}/{target}_{biasA}_{biasB}_{iii}.jpg'
                    img.save(open(fpath, 'w'))

        print('========> finished.')

        # check
        print('========> check')
        self.binary_attribute_count(args.target_id,
                                    args.biasA_id,
                                    args.biasB_id,
                                    attr_matrix=self.attr_matrix[train_indices])


def f():
    gen = CelebAGEN(args=args)
    gen.binary_attribute_count(args.target_id, args.biasA_id, args.biasB_id)


def main():
    gen = CelebAGEN(args=args)
    gen.gen_data()


if __name__ == '__main__':
    args = arg_parse()
    if args.view:
        f()
    else:
        main()
