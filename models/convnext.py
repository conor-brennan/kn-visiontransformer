import torch
from torch import nn
import torchvision.models as models


# LayerNorm is equivalent to GroupNorm with number of groups of 1
ln_layer = lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels)



### Unmodified models (no low-res option) ###

# Unnormalized
def convnext_tiny_nn(num_classes: int = 1000) -> nn.Module:
    return models.convnext_tiny(num_classes=num_classes, norm_layer=nn.Identity)

# LayerNorm
def convnext_tiny_ln(num_classes: int = 1000) -> nn.Module:
    return models.convnext_tiny(num_classes=num_classes, norm_layer=ln_layer)

# BatchNorm
def convnext_tiny_bn(num_classes: int = 1000) -> nn.Module:
    return models.convnext_tiny(num_classes=num_classes, norm_layer=nn.BatchNorm2d)

# GroupNorm
def convnext_tiny_gn(num_classes: int = 1000, num_groups: int = 32) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return models.convnext_tiny(num_classes=num_classes, norm_layer=gn_layer)