"""
Author: Rui Hu
All rights reserved.
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset

from .erm import ERMTrainer
from utils import IdxDataset, AverageMeter


class JTTTrainer(ERMTrainer):
    method_name = "jtt"

    @property
    def run_name(self):
        args = self.args
        if args.bias_label_type == "auto":
            name = f"{self.method_name}_epoch_{args.bias_epoch}_up_weight_{args.jtt_up_weight}" \
                   f"_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        else:
            name = f"{self.method_name}_{args.bias_label_type}_up_weight_{args.jtt_up_weight}" \
                   f"_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        return name

    def init_biased_model(self, train_loader):
        args = self.args
        bias_model_path = self.exp_dir / "bias_model_e5.pth"
        if os.path.exists(bias_model_path):
            print("using existing bias model checkpoint at {}".format(bias_model_path))
            self.model.load_state_dict(torch.load(bias_model_path)["model_state_dict"])
        else:
            # train bias model
            self.model.train()
            losses = AverageMeter()
            pbar = tqdm(range(1, args.bias_epoch + 1))
            for epoch in pbar:
                for _, (image, target) in train_loader:
                    image = image.cuda()
                    label = target[:, 0].cuda()
                    bs = len(image)

                    with torch.cuda.amp.autocast(enabled=args.amp):
                        output, _ = self.model(image)
                        loss = self.criterion(output, label)

                    losses.update(loss.item(), bs)

                    self._loss_backward(loss)
                    self._optimizer_step(self.optimizer)
                    self._scaler_update()
                    self.optimizer.zero_grad(set_to_none=True)

                    pbar.set_description(
                        f"init_biased_model "
                        f"lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:g} "
                        f"loss: {losses.avg:.6f}"
                    )
                self.infer_bias_label()
                self.step_lr.step()
                if epoch % 5 == 0:
                    print("saving bias model")
                    self._save_checkpoint(name=f"bias_model_e{epoch}")
                    self._save_checkpoint(name=f"bias_model_last")

    def _before_train(self):
        args = self.args
        if args.bias_label_type == 'auto':
            self.init_biased_model(self.train_loader)
            global_bias_label = self.infer_bias_label()
        else:
            raise NotImplementedError

        error_indices = torch.where(global_bias_label == 0)[0]
        # up-weight error list
        error_dataset_list = [Subset(self.train_dataset, error_indices)] * args.jtt_up_weight
        error_dataset = ConcatDataset(error_dataset_list)
        new_train_dataset = ConcatDataset([
            self.train_dataset,
            error_dataset
        ])

        self.train_loader = DataLoader(
            IdxDataset(new_train_dataset),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # reinitialize the model and optimizer
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
