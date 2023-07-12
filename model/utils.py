"""
Author: Rui Hu
All rights reserved.
"""

from .resnet import ResNet18, ResNet50


def get_model(arch, num_classes):
    if arch == "resnet18":
        model = ResNet18
    elif arch == "resnet50":
        model = ResNet50
    else:
        raise NotImplementedError

    return model(
        num_classes=num_classes
    )