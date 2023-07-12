"""
Author: Rui Hu
All rights reserved.
"""

import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter


class ERMTrainer(BaseTrainer):
    method_name = "erm"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{args.seed}"
        return name

    def train(self):
        self.model.train()

        args = self.args
        pbar = tqdm(self.train_loader)
        losses = AverageMeter()

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
                output, _ = self.model(image)
                loss = self.criterion(output, label)

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