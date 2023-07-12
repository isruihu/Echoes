"""
Author: Rui Hu
All rights reserved.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .erm import ERMTrainer
from utils import AverageMeter, IdxDataset, EPS
from model.utils import get_model


class DebiANTrainer(ERMTrainer):
    method_name = "debian"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{args.seed}"
        return name

    def _setup_train_loader(self):
        super(DebiANTrainer, self)._setup_train_loader()
        args = self.args
        self.second_train_loader = DataLoader(
            IdxDataset(self.train_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _setup_model(self):
        super()._setup_model()
        args = self.args
        self.bias_discover_net = get_model(
            arch=args.arch,
            num_classes=2,
        ).cuda()

    def _setup_optimizer(self):
        super()._setup_optimizer()
        args = self.args
        self.optimizer_bias_discover_net = torch.optim.Adam(
            params=self.bias_discover_net.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def _train_classifier(self, batch_image, batch_target):
        args = self.args
        self.model.train()
        self.bias_discover_net.eval()

        image, target = batch_image, batch_target

        image = image.cuda()
        label = target[:, 0].cuda()

        with torch.no_grad():
            spurious_logits, _ = self.bias_discover_net(image)
        with torch.cuda.amp.autocast(enabled=args.amp):
            target_logits, _ = self.model(image)

            label = label.reshape(target_logits.shape[0])
            p_vanilla = torch.softmax(target_logits, dim=1)
            p_spurious = torch.sigmoid(spurious_logits)

            ce_loss = self.criterion(target_logits, label)

            # reweight CE with DEO
            for target_val in range(2):
                batch_bool = label.long().flatten() == target_val
                if not batch_bool.any():
                    continue
                p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                positive_spurious_group_avg_p = (
                                                        p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                                                ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                negative_spurious_group_avg_p = (
                                                        (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
                                                ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                if (
                        negative_spurious_group_avg_p
                        < positive_spurious_group_avg_p
                ):
                    p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val

                weight = 1 + p_spurious_w_same_t_val
                ce_loss[batch_bool] *= weight

            ce_loss = ce_loss.mean()

        self._loss_backward(ce_loss)
        self._optimizer_step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)

        return ce_loss.item()

    def _train_bias_discover_net(self, batch_image, batch_target):
        args = self.args
        self.bias_discover_net.train()
        self.model.eval()

        image, target = batch_image, batch_target

        image = image.cuda()
        label = target[:, 0].cuda()

        with torch.no_grad():
            target_logits, _ = self.model(image)
        with torch.cuda.amp.autocast(enabled=args.amp):
            spurious_logits, _ = self.bias_discover_net(image)
            label = label.reshape(target_logits.shape[0])
            p_vanilla = torch.softmax(target_logits, dim=1)
            p_spurious = torch.sigmoid(spurious_logits)

            # ==== deo loss ======
            sum_discover_net_deo_loss = 0
            sum_penalty = 0
            num_classes_in_batch = 0
            for target_val in range(2):
                batch_bool = label.long().flatten() == target_val
                if not batch_bool.any():
                    continue
                p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                positive_spurious_group_avg_p = (
                                                        p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                                                ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                negative_spurious_group_avg_p = (
                                                        (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
                                                ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                discover_net_deo_loss = -torch.log(
                    EPS
                    + torch.abs(
                        positive_spurious_group_avg_p
                        - negative_spurious_group_avg_p
                    )
                )

                negative_p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val
                penalty = -torch.log(
                    EPS
                    + 1
                    - torch.abs(
                        p_spurious_w_same_t_val.mean()
                        - negative_p_spurious_w_same_t_val.mean()
                    )
                )

                sum_discover_net_deo_loss += discover_net_deo_loss
                sum_penalty += penalty
                num_classes_in_batch += 1

            sum_penalty /= num_classes_in_batch
            sum_discover_net_deo_loss /= num_classes_in_batch
            loss_discover = sum_discover_net_deo_loss + sum_penalty

        self._loss_backward(loss_discover)
        self._optimizer_step(self.optimizer_bias_discover_net)
        self.optimizer_bias_discover_net.zero_grad(set_to_none=True)

        return loss_discover.item()

    def train(self):
        args = self.args
        cls_losses = AverageMeter()
        dis_losses = AverageMeter()

        pbar = tqdm(
            zip(self.train_loader, self.second_train_loader),
            dynamic_ncols=True,
            total=len(self.train_loader),
        )

        for (
                (_, main_bacth),
                (_, second_batch),
        ) in pbar:
            main_batch_image, main_batch_target = main_bacth
            second_batch_image, second_batch_target = second_batch

            cls_loss = self._train_classifier(main_batch_image, main_batch_target)
            dis_loss = self._train_bias_discover_net(second_batch_image, second_batch_target)

            cls_losses.update(cls_loss, main_batch_image.size(0))
            dis_losses.update(dis_loss, main_batch_image.size(0))

            self._scaler_update()

        pbar.set_description(
            f"[{self.cur_epoch}/{args.num_epoch}] "
            f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:g} "
            f"cls_loss: {cls_losses.avg:.6f}"
            f"dis_loss: {dis_losses.avg:.6f}"
        )

        self.log_to_wandb({
            "loss": cls_losses.avg,
            "custom_step": self.cur_epoch
        })
