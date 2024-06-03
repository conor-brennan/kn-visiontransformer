# from https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py
# Modifications: Use KNConv2d instead of BatchNorm and Conv2d

import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torchvision.utils import _log_api_usage_once, _make_ntuple

from layers.kernel_norm import KernelNorm2d
from layers.knconv import KNConv2d



class KNConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = 0,           # changed from none
        groups: int = 1,
        # norm_layer: Optional[Callable[..., torch.nn.Module]] = None,       # Changed from BatchNorm, normalized by KNConv2d
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = KNConv2d,
        dropout_p: float = 0.05,
        eps: float = 1e-5,
    ) -> None:

        # if padding is None:
        #     if isinstance(kernel_size, int) and isinstance(dilation, int):
        #         padding = (kernel_size - 1) // 2 * dilation
        #     else:
        #         _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
        #         kernel_size = _make_ntuple(kernel_size, _conv_dim)
        #         dilation = _make_ntuple(dilation, _conv_dim)
        #         padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = False # changed from norm_layer is None

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding, padding) if isinstance(padding, int) else padding
        self.padding = self.padding if len(self.padding) == 4 else (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        self.dropout_p = dropout_p
        self.eps = eps

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                dropout_p=dropout_p,
                eps=eps,
            )
        ]

        # if norm_layer is not None:
        #     layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == KNConvNormActivation:
            warnings.warn(
                "Don't use KNConvNormActivation directly, please use KNConv2dNormActivation and KNConv3dNormActivation instead."
            )


class KNConv2dNormActivation(KNConvNormActivation):
    """
    Configurable block used for KN Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        # norm_layer: Optional[Callable[..., torch.nn.Module]] = None,    # Changed from BatchNorm (already normalized in KNConv2d)
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        dropout_p: float = 0.05,
        eps: float = 1e-5,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            # norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            KNConv2d,
            dropout_p,
            eps,
        )