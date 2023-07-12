"""
Author: Rui Hu
All rights reserved.
"""

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler

from .erm import ERMTrainer
from utils import AverageMeter, IdxDataset


class GroupDROTrainer(ERMTrainer):
    method_name = "groupdro"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        return name

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _setup_train_loader(self):
        args = self.args
        weights = self._get_samping_weights()
        sampler = WeightedRandomSampler(weights, len(self.train_dataset), replacement=True)
        self.train_loader = torch.utils.data.DataLoader(
            IdxDataset(self.train_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=sampler
        )

    def _method_special_setups(self):
        self.group_label = self.train_dataset.group_label
        self.group_num = self.train_dataset.group_num
        self.adv_probs = torch.ones(self.group_num).cuda() / self.group_num
        self.group_range = torch.arange(
            self.group_num, dtype=torch.long
        ).unsqueeze(1)

    def _get_samping_weights(self):
        group_num = self.train_dataset.group_num
        group_label = self.train_dataset.group_label
        group_counts = (
            (torch.arange(group_num).unsqueeze(1) == group_label)
            .sum(1)
            .float()
        )
        group_weights = len(group_label) / group_counts
        print(group_weights)
        weights = group_weights[group_label]
        return weights

    def train(self):
        self.model.train()

        args = self.args
        pbar = tqdm(self.train_loader)
        losses = AverageMeter()

        for indices, (image, target) in pbar:
            image = image.cuda()
            label = target[:, 0].cuda()
            bs = len(image)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output, _ = self.model(image)
                loss_per_sample = self.criterion(output, label)
                # compute group loss
                group_map = (self.group_label[indices] == self.group_range).float().cuda()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count == 0).float()  # avoid nans
                group_loss = (group_map @ loss_per_sample.flatten()) / group_denom
                # update adv_probs
                with torch.no_grad():
                    self.adv_probs = self.adv_probs * torch.exp(
                        args.groupdro_robust_step_size * group_loss.detach()
                    )
                    self.adv_probs = self.adv_probs / (self.adv_probs.sum())
                # compute reweighted robust loss
                loss = group_loss @ self.adv_probs

            losses.update(loss.item(), bs)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            pbar.set_description(
                f"[{self.cur_epoch}/{args.num_epoch}] "
                f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:g} "
                f"loss: {losses.avg:.6f}"
            )

        self.log_to_wandb({
            "loss": losses.avg,
            "custom_step": self.cur_epoch
        })
