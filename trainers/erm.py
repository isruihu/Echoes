"""
Author: Rui Hu
All rights reserved.
"""

import random
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter


class ERMTrainer(BaseTrainer):
    method_name = "erm"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        return name

    # def _setup_train_dataset(self):
    #     super(ERMTrainer, self)._setup_train_dataset()
    #     dataset_size = len(self.train_dataset)
    #     num_samples = int(0.5 * dataset_size)
    #     indices = random.sample(range(dataset_size), num_samples)
    #     self.train_dataset = Subset(self.train_dataset, indices=indices)

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
