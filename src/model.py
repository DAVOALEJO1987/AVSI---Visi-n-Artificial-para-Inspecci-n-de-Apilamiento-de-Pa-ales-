
from __future__ import annotations
from torch import nn
from torchvision import models

def build_resnet18(num_classes: int, pretrained: bool = True, dropout: float = 0.3, freeze_backbone: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))
    return model
