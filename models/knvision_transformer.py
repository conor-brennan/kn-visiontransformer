# from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# Modifications: TODO


import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

import numpy as np
from layers.knconv import KNConv2d
from layers.kernel_norm import KernelNorm2d
from layers.KNConv2dNormActivation import KNConv2dNormActivation


__all__ = [
    "VisionTransformer",
    "knvit_s_16",
    "knvit_s_8",
]


class ConvStemConfigKN(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = KernelNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

def flat_patch_norm(input: torch.Tensor, norm_layer: Callable[..., nn.Module] = KernelNorm2d):
        n, s, e = input.shape
        p = int(np.sqrt(e))
        # print("FPN input shape", input.shape)
        # print("FPN input type", input.dtype)
        x = input.reshape(n, s, p, p)
        # print("FPN x shape", x.shape)
        # print("FPN x type", x.dtype)
        x = norm_layer(x)
        # print("FPN x shape again", x.shape)
        # print("FPN x type again", x.dtype)
        # n, s, h, w = x.shape
        x = x.reshape(n, s, e)
        # print("FPN x return shape", x.shape)
        # print("FPN x return type", x.dtype)
        return x

class EncoderBlockKN(nn.Module):
    """Transformer encoder block with KN."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        # norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # KernelNorm replacing LayerNorm 
        norm_layer = KernelNorm2d
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(kernel_size=2, stride=2)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(kernel_size=2, stride=2)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # Transform (N,S,E) input to work in KernelNorm
        # print('ln1')
        x = flat_patch_norm(input=input, norm_layer=self.ln_1)
        # print("typex, ", x.dtype)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        # print("typex sfa, ", x.dtype)
        x = self.dropout(x)
        # print("typex dp, ", x.dtype)
        x = x + input
        # print("typex inp, ", x.dtype)
        # print('ln2')
        y = flat_patch_norm(input=x, norm_layer=self.ln_2)
        y = self.mlp(y)
        return x + y
    
    # def forward(self, input: torch.Tensor):
    #     torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    #     # Transform (N,S,E) input to work in KernelNorm
    #     print('ln1')
    #     # TODO: unpatchify
    #     x = self.ln_1(input)
    #     # TODO: repatchify
    #     x, _ = self.self_attention(x, x, x, need_weights=False)
    #     x = self.dropout(x)
    #     x = x + input
    #     print('ln2')
    #     # TODO: unpatchify
    #     y = self.ln_2(x)
    #     # TODO: repatchify
    #     y = self.mlp(y)
    #     return x + y    

class EncoderKN(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        # norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # KernelNorm replacing LayerNorm 
        norm_layer = KernelNorm2d
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlockKN(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(kernel_size=2, stride=2)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        # print('encoder ln')
        layers = self.layers(self.dropout(input))
        # print(f'Layer type: {type(layers)}, dtype: {layers.dtype}, shape: {layers.shape}')
        x = flat_patch_norm(input=layers, norm_layer=self.ln)
        return x
    
    # def forward(self, input: torch.Tensor):
    #     torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    #     input = input + self.pos_embedding
    #     print('encoder ln')
    #     return self.ln(self.layers(self.dropout(input)))


class KNVisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        # norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfigKN]] = None,
    ):
        super().__init__()
        # KernelNorm replacing LayerNorm 
        norm_layer = KernelNorm2d
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    KNConv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        # norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", KNConv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = KNConv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = EncoderKN(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem\
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # print("n,c,h,w ",n,c,h,w)
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        # print("nh,nw ",n_h,n_w)

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # print("after conx", x.shape)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        # print("n,s,e", x.shape)
        # print("OG X TYPING", x.dtype)
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # print('self encoder')
        x = self.encoder(x)
        # print('passed the self encoder')
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def _knvision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    progress: bool,
    weights: Optional[WeightsEnum] = None,
    **kwargs: Any,
) -> KNVisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = KNVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model



@register_model()
def knvit_s_16(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> KNVisionTransformer:
    """
    Constructs a knvit_s_16 architecture from
    Args:
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``KNVisionTransformer`` base class.
    """
    # made the hidden dim square to support KernelNorm (more details in EncoderBlockKN) 
    return _knvision_transformer(
        patch_size=16,
        num_layers=16,
        num_heads=14,
        hidden_dim=784,
        mlp_dim=3072,
        progress=progress,
        image_size=image_size,
        **kwargs,
    )

@register_model()
def knvit_s_8(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> KNVisionTransformer:
    """
    Constructs a knvit_s_8 architecture from
    Args:
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``KNVisionTransformer`` base class.
    """
    # made the hidden dim square to support KernelNorm (more details in EncoderBlockKN) 
    return _knvision_transformer(
        patch_size=8,
        num_layers=16,
        num_heads=14,
        hidden_dim=784,
        mlp_dim=3072,
        progress=progress,
        image_size=image_size,
        **kwargs,
    )


# @register_model()
# def knvit_b_16(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> KNVisionTransformer:
#     """
#     Constructs a knvit_b_16 architecture from
#     Args:
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``KNVisionTransformer`` base class.
#     """

#     return _knvision_transformer(
#         patch_size=16,
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072,
#         progress=progress,
#         image_size=image_size,
#         **kwargs,
#     )

# @register_model()
# def knvit_b_32(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> KNVisionTransformer:
#     """
#     Constructs a knvit_b_32 architecture from
#     Args:
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``KNVisionTransformer`` base class.
#     """

#     return _knvision_transformer(
#         patch_size=32,
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072,
#         progress=progress,
#         image_size=image_size,
#         **kwargs,
#     )

# @register_model()
# def knvit_l_16(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> KNVisionTransformer:
#     """
#     Constructs a knvit_l_16 architecture from
#     Args:
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``KNVisionTransformer`` base class.
#     """

#     return _knvision_transformer(
#         patch_size=16,
#         num_layers=24,
#         num_heads=16,
#         hidden_dim=1024,
#         mlp_dim=4096,
#         progress=progress,
#         image_size=image_size,
#         **kwargs,
#     )

# @register_model()
# def knvit_l_32(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> KNVisionTransformer:
#     """
#     Constructs a knvit_l_32 architecture from
#     Args:
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``KNVisionTransformer`` base class.
#     """

#     return _knvision_transformer(
#         patch_size=32,
#         num_layers=24,
#         num_heads=16,
#         hidden_dim=1024,
#         mlp_dim=4096,
#         progress=progress,
#         image_size=image_size,
#         **kwargs,
#     )


# @register_model()
# def knvit_h_14(*, progress: bool = True, image_size: int = 32, **kwargs: Any) -> KNVisionTransformer:
#     """
#     Constructs a knvit_h_14 architecture from
#     Args:
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``KNVisionTransformer`` base class.
#     """

#     return _knvision_transformer(
#         patch_size=14,
#         num_layers=32,
#         num_heads=16,
#         hidden_dim=1280,
#         mlp_dim=5120,
#         progress=progress,
#         image_size=image_size,
#         **kwargs,
#     )


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state