"""
Author: Rui Hu
All rights reserved.
"""

import os
from abc import abstractmethod

import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_dataset
from model.utils import get_model
from utils import AverageMeter, IdxDataset


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        # data dir
        self.data_dir = Path(args.root)
        self.exp_dir = Path(args.exp_dir) / args.dataset / self.run_name

        self.is_best = False
        self.cur_epoch = 1
        self.best_epoch = 1
        self.best_val_log_dict = {}
        self.best_test_log_dict = {}
        self.con_test_log_dict = {}

        print(f"Run Name: {self.run_name}")

    @property
    @abstractmethod
    def run_name(self):
        pass

    def _setup_all(self):
        args = self.args
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        self._setup_datasets()
        self.num_classes = self.train_dataset.num_classes
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_wandb()
        self._method_special_setups()

    def _setup_datasets(self):
        args = self.args

        self._setup_train_dataset()
        val_dataset = get_dataset(
            dataset_name=args.dataset,
            split='val',
            args=self.args
        )
        test_dataset = get_dataset(
            dataset_name=args.dataset,
            split='test',
            args=self.args
        )
        self._setup_train_loader()
        self.val_loader = DataLoader(
            IdxDataset(val_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            IdxDataset(test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        print(f"Dataset:"
              f"\n  train dataset size: {len(self.train_dataset)}"
              f"\n    val dataset size: {len(val_dataset)}"
              f"\n   test dataset size: {len(test_dataset)}")

    def _setup_model(self):
        args = self.args
        model = get_model(
            arch=args.arch,
            num_classes=self.num_classes
        )
        self.model = model.cuda()
        print(f"Model:\n  {args.arch}")

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        args = self.args
        self.optimizer = optim.Adam(
            self.model.parameters(),
            args.lr,
            weight_decay=args.weight_decay
        )

    def _setup_scheduler(self):
        step_size = int(self.args.num_epoch / 10)
        if step_size == 0:
            step_size += 1
        self.step_lr = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=0.65
        )

    def _setup_wandb(self, name=None):
        args = self.args
        if name is None:
            name = self.run_name
        if args.wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=name,
                config=args,
                settings=wandb.Settings(start_method="fork"),
            )

    def _setup_train_dataset(self):
        args = self.args
        self.train_dataset = get_dataset(
            dataset_name=args.dataset,
            split='train',
            args=args
        )

    def _setup_train_loader(self):
        args = self.args
        self.train_loader = DataLoader(
            IdxDataset(self.train_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    def _save_checkpoint(self, name=None, model=None):
        if model is None:
            model = self.model
        if name is None:
            name = "last"
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.cur_epoch,
        }
        ckpt_fpath = self.exp_dir
        os.makedirs(ckpt_fpath, exist_ok=True)
        torch.save(ckpt, Path(ckpt_fpath) / f'{name}.pth')

    def _method_special_setups(self):
        pass

    def _before_train(self):
        pass

    def _after_train(self):
        pass

    def _batch_transform(
            self,
            batch_data,
            batch_label,
            batch_indices
    ):
        """
        Transform batch data in training

        :param batch:
        :return:
        """
        return batch_data, batch_label

    def _loss_backward(self, loss, retain_graph=False):
        if self.args.amp:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    @abstractmethod
    def train(self):
        pass

    @torch.no_grad()
    def infer_bias_label(self, model=None) -> torch.LongTensor:
        """training set: Correct as bias-aligned (bias=1), error as bias-conflicted (bias=0)"""
        args = self.args
        # get bias label
        bias_label_list = []

        if model is None:
            model = self.model
        model.eval()

        unshuffled_train_loader = DataLoader(
            IdxDataset(self.train_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )

        group_num_list = [{'total': 0, 'error': 0} for _ in range(self.train_dataset.group_num)]
        for indices, (image, target) in unshuffled_train_loader:
            image = image.cuda()
            with torch.cuda.amp.autocast(enabled=args.amp):
                output, _ = model(image)

            pred = output.argmax(dim=1).cpu()
            label = target[:, 0]

            bias_label = (pred == label).long()
            bias_label_list.append(bias_label)

            error = (pred != label).bool()
            batch_group_label = self.train_dataset.group_label[indices]
            for i in range(self.train_dataset.group_num):
                group_indices = torch.where(batch_group_label == i)[0]
                group_num_list[i]['total'] += len(group_indices)
                group_num_list[i]['error'] += torch.sum(error[group_indices])

        for i, dic in enumerate(group_num_list):
            print(f"group{i}: {dic['error']}/{dic['total']} ", end='')
            if i == 1 or i == 3: print("")

        bias_label = torch.concat(bias_label_list, dim=0)
        return bias_label

    @torch.no_grad()
    def _eval_split(self, split='val'):
        args = self.args

        if split == 'val':
            loader = self.val_loader
        else:
            loader = self.test_loader

        group_label = loader.dataset.dataset.group_label

        accuracy_dict = {
            f"group_{i}": AverageMeter() for i in range(self.train_dataset.group_num)
        }
        accuracy_dict.update({
            "bias_aligned": AverageMeter(),
            "bias_conflicted": AverageMeter(),
            "normal": AverageMeter()
        })

        self.model.eval()
        pbar = tqdm(loader, dynamic_ncols=True, leave=False)
        for indices, (image, target) in pbar:
            image = image.cuda()
            bs = len(image)
            batch_group_label = group_label[indices]

            with torch.cuda.amp.autocast(enabled=args.amp):
                output, _ = self.model(image)

            pred = output.argmax(dim=1).cpu()

            task_label = target[:, 0]
            is_aligned_label = target[:, 1]

            task_correct = (pred == task_label).bool()
            accuracy_dict["normal"].update(
                correct_num=torch.sum(task_correct).item(),
                batch_size=bs
            )

            bias_aligned_indices = torch.where(is_aligned_label == 1)[0]
            bias_conflicted_indices = torch.where(is_aligned_label == 0)[0]
            group_indices_list = [
                torch.where(batch_group_label == i)[0] for i in range(4)
            ]

            if len(bias_aligned_indices):
                accuracy_dict["bias_aligned"].update(
                    correct_num=torch.sum(task_correct[bias_aligned_indices]).item(),
                    batch_size=len(bias_aligned_indices)
                )
            if len(bias_conflicted_indices):
                accuracy_dict["bias_conflicted"].update(
                    correct_num=torch.sum(task_correct[bias_conflicted_indices]).item(),
                    batch_size=len(bias_conflicted_indices)
                )
            for i in range(4):
                if len(group_indices_list[i]):
                    accuracy_dict[f"group_{i}"].update(
                        correct_num=torch.sum(task_correct[group_indices_list[i]]).item(),
                        batch_size=len(group_indices_list[i])
                    )

        log_dict = self._build_log_dict(
            split=split,
            accuracy_dict=accuracy_dict)
        return log_dict

    def _build_log_dict(self, split, accuracy_dict):
        args = self.args
        log_dict = {}
        for k, v in accuracy_dict.items():
            log_dict[f"{split}_{k}_acc"] = v.avg
        log_dict.update({
            f"{split}_unbiased_acc": (accuracy_dict["bias_aligned"].avg +
                                      accuracy_dict["bias_conflicted"].avg) / 2,
            f"{split}_worst_group_acc": min(accuracy_dict["group_0"].avg,
                                            accuracy_dict["group_1"].avg,
                                            accuracy_dict["group_2"].avg,
                                            accuracy_dict["group_3"].avg),
            f"{split}_group_avg_acc": np.average([accuracy_dict["group_0"].avg,
                                                  accuracy_dict["group_1"].avg,
                                                  accuracy_dict["group_2"].avg,
                                                  accuracy_dict["group_3"].avg]),
        })

        for key in log_dict.keys():
            log_dict[key] *= 100
        return log_dict

    def eval(self):
        args = self.args
        val_log_dict = self._eval_split('val')
        test_log_dict = self._eval_split('test')

        # check whether is best
        compare_key = f"val_{args.model_selection_meter}"
        for key, value in val_log_dict.items():
            new_key = f"best_{key}"
            if (
                    new_key not in self.best_val_log_dict.keys() or
                    value >= self.best_val_log_dict[new_key]
            ):
                self.best_val_log_dict[new_key] = value
                if key == compare_key:
                    self.is_best = True
                    self.con_test_log_dict = {
                        f"con_{k}": v for k, v in test_log_dict.items()
                    }
                    self.best_epoch = self.cur_epoch

        print("***************************** group acc ************************************")
        print("val:  ", end="")
        for i in range(self.train_dataset.group_num):
            if i == 2: print("      ", end="")
            print(f"group{i} acc {val_log_dict[f'val_group_{i}_acc']:.1f}", end=" ")
            if i == 1 or i == 3: print("")
        print("test: ", end="")
        for i in range(self.train_dataset.group_num):
            if i == 2: print("      ", end="")
            print(f"group{i} acc {test_log_dict[f'test_group_{i}_acc']:.1f}", end=" ")
            if i == 1 or i == 3: print("")
        print("**************************** test metrics *********************************")
        print("current test")
        print(f"test worst group acc {test_log_dict['test_worst_group_acc']:.1f} || "
              f"test group avg acc {test_log_dict['test_group_avg_acc']:.1f} || "
              f"test bias-aligned acc {test_log_dict['test_bias_aligned_acc']:.1f} || "
              f"test bias-conflicted acc {test_log_dict['test_bias_conflicted_acc']:.1f}")
        print(f"Best epoch: {self.best_epoch}    (by {args.model_selection_meter})")
        print(f"test worst group acc {self.con_test_log_dict['con_test_worst_group_acc']:.1f} || "
              f"test group avg acc {self.con_test_log_dict['con_test_group_avg_acc']:.1f} || "
              f"test bias-aligned acc {self.con_test_log_dict['con_test_bias_aligned_acc']:.1f} || "
              f"test bias-conflicted acc {self.con_test_log_dict['con_test_bias_conflicted_acc']:.1f}")
        print("***************************************************************************\n")

        # add custom step
        val_log_dict.update({
            "custom_step": self.cur_epoch
        })
        test_log_dict.update({
            "custom_step": self.cur_epoch
        })
        self.best_val_log_dict.update({
            "custom_step": self.cur_epoch
        })
        self.best_test_log_dict.update({
            "custom_step": self.cur_epoch
        })

        self.log_to_wandb(val_log_dict)
        self.log_to_wandb(test_log_dict)
        self.log_to_wandb(self.best_val_log_dict)
        self.log_to_wandb(self.best_test_log_dict)

    def log_to_wandb(self, log_dict):
        assert "custom_step" in log_dict.keys()
        if self.args.wandb:
            wandb.log(log_dict)

    def __call__(self):
        args = self.args
        self._setup_all()
        self._before_train()
        for e in range(1, args.num_epoch + 1):
            print(f"Epoch {self.cur_epoch}")
            self.train()
            self._after_train()
            self.step_lr.step()
            if args.not_eval is False:
                self.eval()
            if args.save_ckpt:
                self._save_checkpoint()
            self.cur_epoch += 1
