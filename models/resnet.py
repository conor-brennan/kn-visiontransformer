import torch
from torch import nn
import torchvision.models as models

### Unmodified models (no low-res option) ###

# Unnormalized
def resnet18_nn(num_classes: int = 1000) -> nn.Module:
    return models.resnet18(num_classes=num_classes, norm_layer=nn.Identity)

# LayerNorm
def resnet18_ln(num_classes: int = 1000) -> nn.Module:
    return models.resnet18(num_classes=num_classes, norm_layer=nn.LayerNorm)

# BatchNorm
def resnet18_bn(num_classes: int = 1000) -> nn.Module:
    return models.resnet18(num_classes=num_classes, norm_layer=nn.BatchNorm2d)

# GroupNorm
def resnet18_gn(num_classes: int = 1000, num_groups: int = 32) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return models.resnet18(num_classes=num_classes, norm_layer=gn_layer)