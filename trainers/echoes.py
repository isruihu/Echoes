"""
Author: Rui Hu
All rights reserved.
"""

import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter
from model.utils import get_model


class EchoesTrainer(BaseTrainer):
    method_name = "echoes"

    @property
    def run_name(self):
        args = self.args
        if args.sigmoid:
            name = f"{self.method_name}_alpha_{args.alpha}_threshold_{args.t_error}_lambda_{int(args.lambda_)}" \
                   f"_sigmoid_{args.sigmoid}" \
                   f"_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        else:
            name = f"{self.method_name}_alpha_{args.alpha}_threshold_{args.t_error}_lambda_{int(args.lambda_)}" \
                   f"_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        return name

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _setup_model(self):
        self.model = get_model(self.args.arch, num_classes=self.num_classes)
        self.bias_model = get_model(self.args.arch, num_classes=self.num_classes)
        self.model = self.model.cuda()
        self.bias_model = self.bias_model.cuda()

    def _setup_optimizer(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)
        self.bias_optimizer = torch.optim.Adam(params=self.bias_model.parameters(),
                                                  lr=self.args.lr,
                                                  weight_decay=self.args.weight_decay)

    def _setup_scheduler(self):
        step_size=10
        self.step_lr = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=0.65
        )
        self.bias_step_lr = torch.optim.lr_scheduler.StepLR(
            self.bias_optimizer,
            step_size=step_size,
            gamma=0.65
        )

    def _method_special_setups(self):
        self.weight = torch.zeros(len(self.train_dataset)).cuda()
        self.bias_weight = torch.ones(len(self.train_dataset)).cuda()
        self.task_label = torch.as_tensor(self.train_dataset.target[:, 0])

    def train(self):
        self.model.train()

        args = self.args
        pbar = tqdm(self.train_loader)
        losses = AverageMeter()

        total_bias_loss = 0
        total_debias_loss = 0
        iter = 1

        for indices, (image, target) in pbar:
            image = image.cuda()
            label = target[:, 0].cuda()
            bs = len(image)

            image, label = self._batch_transform(
                batch_data=image,
                batch_label=label,
                batch_indices=indices
            )

            with torch.cuda.amp.autocast(enabled=args.amp):
                bias_output, _ = self.bias_model(image)
                bias_loss = self.bias_criterion(bias_output, label)
                bias_loss = (bias_loss * self.bias_weight[indices]).mean()

                debias_output, _ = self.model(image)
                debias_loss = self.criterion(debias_output, label)
                debias_loss = (debias_loss * self.weight[indices]).mean()

            total_bias_loss += bias_loss.item()
            total_debias_loss += debias_loss.item()

            loss = bias_loss + args.lambda_ * debias_loss
            losses.update(loss.item(), bs)

            self.optimizer.zero_grad(set_to_none=True)
            self.bias_optimizer.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._optimizer_step(self.bias_optimizer)
            self._scaler_update()

            pbar.set_description(
                f"[{self.cur_epoch}/{args.num_epoch}] "
                f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:g} "
                f"loss: {losses.avg:.6f} "
                f"bias loss {total_bias_loss / iter:.6f} "
                f"debias loss {total_debias_loss / iter:.6f}"
            )
            iter += 1

        self.log_to_wandb({
            "loss": losses.avg,
            "custom_step": self.cur_epoch
        })

    def _after_train(self):
        args = self.args
        num_sample = len(self.train_dataset)
        is_correct = self.infer_bias_label(self.bias_model)

        # bias weight
        for c in range(self.num_classes):
            class_num = torch.sum(self.task_label == c)
            class_error_num = torch.sum(is_correct[torch.where(self.task_label == c)] == 0)
            if class_error_num < (class_num * args.t_error):
                self.bias_weight[(is_correct == 0) & (self.task_label == c)] *= args.alpha
                self.bias_weight /= (self.bias_weight.max() + 1e-8)

        # debias weight
        if args.sigmoid:
            self.weight = torch.sigmoid(-1 * self.bias_weight)
        else:
            self.weight = 1 / (self.bias_weight + 1e-8)
        # to make class weight balanced
        if args.not_class_balance is False:
            weight_sum_per_class = [torch.sum(self.weight[self.task_label == i]) for i in range(self.num_classes)]
            weight_product = 1
            for w in weight_sum_per_class:
                weight_product *= w
            for i, w in enumerate(weight_sum_per_class):
                magnification = (weight_product / w)
                self.weight[self.task_label == i] *= magnification
        # normalization
        self.weight /= (self.weight.max() + 1e-8)

        # save weights
        # os.makedirs(self.exp_dir, exist_ok=True)
        # torch.save(self.weight.cpu(), Path(self.exp_dir) / f'weight_{self.cur_epoch}.pth')

    def __call__(self):
        args = self.args
        self._setup_all()
        self._before_train()
        for e in range(1, args.num_epoch + 1):
            print(f"{args.method} Epoch {self.cur_epoch}")
            self.train()
            self._after_train()
            self.bias_step_lr.step()
            self.step_lr.step()
            if args.not_eval is False:
                self.eval()
            if args.save_ckpt:
                self._save_checkpoint()
            self.cur_epoch += 1