"""
Author: Rui Hu
All rights reserved.
"""

import torch.nn as nn
from torchvision.models import resnet18, resnet50


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = resnet18()
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x):
        feat = self.extract(x)
        logits = self.fc(feat)
        return logits, feat

    def extract(self, x):
        feat = self.extractor(x).squeeze(-1).squeeze(-1)
        return feat


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = resnet50()
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 2048
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x):
        feat = self.extract(x)
        logits = self.fc(feat)
        return logits, feat

    def extract(self, x):
        feat = self.extractor(x).squeeze(-1).squeeze(-1)
        return feat

