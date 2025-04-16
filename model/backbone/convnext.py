import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from typing import Sequence
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
@MODELS.register_module()
class ConvNeXt(BaseModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    arch_settings = {
        "tiny": {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        "small": {"depths": [3, 3, 27, 3], "channels": [96, 192, 384, 768]},
        "base": {"depths": [3, 3, 27, 3], "channels": [128, 256, 512, 1024]},
        "large": {"depths": [3, 3, 27, 3], "channels": [192, 384, 768, 1536]},
    }
    def __init__(
        self,
        arch="tiny",
        in_chans=3,
        drop_path_rate=0., 
        layer_scale_init_value=1e-6, head_init_scale=1.,
        init_cfg=None,
        out_indices=-1,
        norm_out=False,
        frozen_stages=0,
        with_cp=False,
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, (
                f"Unavailable arch, please choose from "
                f"({set(self.arch_settings)}) or pass a dict."
            )
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert "depths" in arch and "channels" in arch, (
                f'The arch dict must have "depths" and "channels", '
                f"but got {list(arch.keys())}."
            )

        self.depths = arch["depths"]
        self.channels = arch["channels"]
        assert (
            isinstance(self.depths, Sequence)
            and isinstance(self.channels, Sequence)
            and len(self.depths) == len(self.channels)
        ), (
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) '
            "should be both sequence with the same length."
        )

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), (
            f'"out_indices" must by a sequence or int, '
            f"get {type(out_indices)} instead."
        )
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f"Invalid out_indices {index}"
        self.out_indices = out_indices
        self.norm_out = norm_out
        self.frozen_stages = frozen_stages
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(
                in_chans, 
                self.channels[0],
                kernel_size=4,
                stride=4
            ),
            LayerNorm(self.channels[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(self.channels[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[Block(dim=self.channels[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

            if i in self.out_indices and self.norm_out:
                norm_layer = LayerNorm(self.channels[i], eps=1e-6, data_format="channels_first")
                self.add_module(f"norm{i}", norm_layer)

        self.apply(self._init_weights)
        self._freeze_stages()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0., std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                out_x = getattr(self, f"norm{i}")(x) if self.norm_out else x
                outs.append(out_x)

        return tuple(outs)
    
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(), stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x