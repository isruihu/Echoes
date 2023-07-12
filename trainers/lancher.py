"""
Author: Rui Hu
All rights reserved.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
from trainers import *
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            'celeba',
            'urbancars'
        ],
        required=True
    )
    parser.add_argument("--root", type=str, default="data/")
    # celeba
    parser.add_argument("--target_id", type=int, default=1)
    parser.add_argument("--biasA_id", type=int, default=20)
    parser.add_argument("--biasB_id", type=int, default=39)
    parser.add_argument("--biasA_ratio", type=float, default=0.95)
    parser.add_argument("--biasB_ratio", type=float, default=0.95)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--method", required=True)
    parser.add_argument("--arch", type=str, default='resnet18')
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument(
        "--model_selection_meter",
        type=str,
        choices=[
            'worst_group_acc',
            'normal_acc',
            'id_acc'
        ],
        default='worst_group_acc'
    )
    parser.add_argument("--exp_dir", type=str, default="exp/")
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--not_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_offline", action="store_true")

    # JTT
    parser.add_argument("--bias_epoch", type=int, default=1)
    parser.add_argument("--jtt_up_weight", type=int, default=50)
    parser.add_argument(
        "--bias_label_type",
        type=str,
        choices=[
            "biasA",
            "biasB",
            "auto",
        ],
        default="auto"
    )
    # GroupDRO
    parser.add_argument("--groupdro_robust_step_size", type=float, default=0.01)
    parser.add_argument("--groupdro_gamma", type=float, default=0.1)
    # Echoes
    parser.add_argument("--t_error", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_", type=float, default=1000)
    parser.add_argument("--not_class_balance", action="store_true")
    parser.add_argument("--sigmoid", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    print(f'=================================> Using Fixed Random Seed: {args.seed} <=================================')

    if args.wandb:
        args.wandb_project_name = args.dataset
        args.wandb_entity = None  # Your entity name
        assert args.wandb_entity is not None
        if args.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'

    return args


if __name__ == '__main__':
    args = parse_args()
    method = methods[
        args.method
    ](args)
    print("Method: {}".format(method.method_name))
    method()
