"""
Author: Rui Hu
All rights reserved.
"""

from .erm import ERMTrainer
from model.criterion import GeneralizedCELoss


class GCETrainer(ERMTrainer):
    method_name = "gce"

    @property
    def run_name(self):
        args = self.args
        name = f"{self.method_name}_{self.target_name}_{self.biasA_name}_{self.biasB_name}_{args.seed}"
        return name

    def _before_train(self):
        self.criterion = GeneralizedCELoss()
