import torch
from torch import nn
import torchvision.models as models

from torchvision.models.vision_transformer import _vision_transformer, VisionTransformer
from typing import Any, Callable, Dict, List, NamedTuple, Optional

### Unmodified models (no low-res option) ###

# Unnormalized
def vit_b_16_nn(num_classes: int = 1000) -> nn.Module:
    return models.vit_b_16(num_classes=num_classes, norm_layer=nn.Identity, image_size=32)

# LayerNorm
def vit_b_16_ln(num_classes: int = 1000) -> nn.Module:
    return models.vit_b_16(num_classes=num_classes, norm_layer=nn.LayerNorm, image_size=32)

# BatchNorm
def vit_b_16_bn(num_classes: int = 1000) -> nn.Module:
    return models.vit_b_16(num_classes=num_classes, norm_layer=nn.BatchNorm2d, image_size=32)

# GroupNorm
def vit_b_16_gn(num_classes: int = 1000, num_groups: int = 32) -> nn.Module:
    gn_layer = lambda num_in_channels: nn.GroupNorm(num_groups=num_groups, num_channels=num_in_channels)
    return models.vit_b_16(num_classes=num_classes, norm_layer=gn_layer, image_size=32)

# Same size as KNViT-S-16 (ish)
def vit_s_16(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> VisionTransformer:
    # made the hidden dim square to match supported KernelNorm (more details in EncoderBlockKN) 
    return _vision_transformer(
        patch_size=16,
        num_layers=16,
        num_heads=14,
        hidden_dim=784,
        mlp_dim=3072,
        weights = None,
        progress=progress,
        image_size=image_size,
        **kwargs,
    )

# Smaller patch size, maintains square hidden dim
def vit_s_8(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> VisionTransformer:
    # made the hidden dim square to match supported KernelNorm (more details in EncoderBlockKN) 
    return _vision_transformer(
        patch_size=8,
        num_layers=16,
        num_heads=14,
        hidden_dim=784,
        mlp_dim=3072,
        weights = None,
        progress=progress,
        image_size=image_size,
        **kwargs,
    )