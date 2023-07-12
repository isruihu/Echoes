"""
Author: Rui Hu
All rights reserved.
"""

import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from model.utils import get_model
from model.criterion import GeneralizedCELoss
from utils import EMAGPU as EMA


class LfFTrainer(BaseTrainer):
    method_name = "lff"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{args.seed}"
        return name

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.gce_criterion = GeneralizedCELoss()

    def _setup_model(self):
        args = self.args
        super(LfFTrainer, self)._setup_model()
        self.bias_model = get_model(
            arch=args.arch,
            num_classes=self.num_classes
        ).cuda()

    def _setup_optimizer(self):
        super(LfFTrainer, self)._setup_optimizer()
        self.bias_optimizer = torch.optim.Adam(
            params=self.bias_model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

    def _method_special_setups(self):
        train_label = self.train_dataset.target[:, 0]
        self.sample_loss_ema_b = EMA(
            torch.LongTensor(train_label), alpha=0.7
        )
        self.sample_loss_ema_d = EMA(
            torch.LongTensor(train_label), alpha=0.7
        )

    def train(self):
        args = self.args
        self.model.train()
        self.bias_model.train()

        total_cls_loss = 0
        total_ce_loss = 0
        total_gce_loss = 0

        pbar = tqdm(self.train_loader)
        for idx, (batch_indices, (image, target)) in enumerate(pbar):
            image = image.cuda()
            label = target[:, 0].cuda()

            with torch.cuda.amp.autocast(enabled=args.amp):
                target_logits, _ = self.model(image)
                spurious_logits, _ = self.bias_model(image)

                ce_loss = self.criterion(target_logits, label)
                gce_loss = self.gce_criterion(spurious_logits, label).mean()

            loss_b = self.criterion(spurious_logits, label).detach()
            loss_d = ce_loss.detach()

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, batch_indices)
            self.sample_loss_ema_d.update(loss_d, batch_indices)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[batch_indices].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[batch_indices].clone().detach()

            max_loss_b = self.sample_loss_ema_b.max_loss(label)
            max_loss_d = self.sample_loss_ema_d.max_loss(label)
            loss_b /= max_loss_b
            loss_d /= max_loss_d

            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            ce_loss = (ce_loss * loss_weight).mean()

            loss = ce_loss + gce_loss

            self.optimizer.zero_grad(set_to_none=True)
            self.bias_optimizer.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._optimizer_step(self.bias_optimizer)

            self._scaler_update()

            total_cls_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_gce_loss += gce_loss.item()
            avg_cls_loss = total_cls_loss / (idx + 1)
            avg_ce_loss = total_ce_loss / (idx + 1)
            avg_gce_loss = total_gce_loss / (idx + 1)

            pbar.set_description(
                "[{}/{}] cls_loss: {:.3f}, ce: {:.3f}, gce: {:.3f}".format(
                    self.cur_epoch,
                    args.num_epoch,
                    avg_cls_loss,
                    avg_ce_loss,
                    avg_gce_loss,
                )
            )

        log_dict = {
            "loss": total_cls_loss / len(self.train_loader),
            "ce_loss": total_ce_loss / len(self.train_loader),
            "gce_loss": total_gce_loss / len(self.train_loader),
            "custom_step": self.cur_epoch
        }
        self.log_to_wandb(log_dict)
