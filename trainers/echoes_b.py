"""
Author: Rui Hu
All rights reserved.
"""

import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter


class EchoesBiasedModelTrainer(BaseTrainer):
    method_name = "echoes_b"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        return name

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _method_special_setups(self):
        self.bias_weight = torch.ones(len(self.train_dataset)).cuda()
        self.task_label = torch.as_tensor(self.train_dataset.target[:, 0])

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
                loss = (loss * self.bias_weight[indices]).mean()

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

    def _after_train(self):
        args = self.args
        num_sample = len(self.train_dataset)
        is_correct = self.infer_bias_label()

        # bias weight
        for c in range(self.num_classes):
            class_num = torch.sum(self.task_label == c)
            class_error_num = torch.sum(is_correct[torch.where(self.task_label == c)] == 0)
            if class_error_num < (class_num * args.t_error):
                self.bias_weight[(is_correct == 0) & (self.task_label == c)] *= args.alpha
                self.bias_weight /= (self.bias_weight.max() + 1e-8)
