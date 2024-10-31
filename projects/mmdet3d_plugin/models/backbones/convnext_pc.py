# Copyright (c) OpenMMLab. All rights reserved.
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import build_norm_layer
from mmcv.runner.base_module import ModuleList, Sequential

from mmdet.models import BACKBONES
from mmcls.models.backbones import ConvNeXt
from mmcls.models.backbones.convnext import ConvNeXtBlock


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            # x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            # self.bias, self.eps).permute(0, 3, 1, 2)
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()  # NOTE


class ConvNeXtBlockLarge(ConvNeXtBlock):
    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 kernel_size=9,
                 padding=4,
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super(ConvNeXtBlockLarge, self).__init__(
            in_channels=in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            mlp_ratio=mlp_ratio,
            linear_pw_conv=linear_pw_conv,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels)


@BACKBONES.register_module()
class ConvNeXt_PC(ConvNeXt):
    arch_settings = {
        'tiny': {
            'depths': [2, 2, 1, 1, 1],
            'channels': [48, 96, 96, 96, 96]
        },
        'small': {
            'depths': [3, 3, 2, 1, 1],
            'channels': [48, 192, 192, 192, 192]
        },
        'base': {
            'depths': [4, 4, 2, 2, 1],
            'channels': [64, 192, 384, 384, 384]
        },
        'large': {
            'depths': [6, 6, 4, 2, 1],
            'channels': [96, 192, 384, 384, 384]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 first_downsample=1,
                 large_arch=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.first_downsample = first_downsample

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        if self.first_downsample == 0:
            self.downsample_layers = ModuleList()
        else:
            self.downsample_layers = ModuleList([None])

        self.bias = torch.nn.Parameter(torch.randn(3))

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()
        if large_arch is not None:
            large_stages = large_arch.get("stages")
            large_kernel_sizes = large_arch.get("large_kernel_sizes")
            large_kernel_paddings = [i // 2 for i in large_kernel_sizes]

        for i in range(self.num_stages):
            depth = self.depths[i]
            if i == 0:
                self.channels[i] = in_channels  # NOTE
            channels = self.channels[i]

            if i >= self.first_downsample:
                if self.first_downsample == 0 and i == 0:
                    downsample_layer = nn.Sequential(
                        LayerNorm2d(in_channels),
                        nn.Conv2d(
                            in_channels,
                            channels,
                            kernel_size=2,
                            stride=2),
                    )
                    self.downsample_layers.append(downsample_layer)
                else:
                    downsample_layer = nn.Sequential(
                        LayerNorm2d(self.channels[i - 1]),
                        nn.Conv2d(
                            self.channels[i - 1],
                            channels,
                            kernel_size=2,
                            stride=2),
                    )
                    self.downsample_layers.append(downsample_layer)

            if large_arch is not None and i in large_stages:
                stage_idx = large_stages.index(3)
                large_kernel_size = large_kernel_sizes[stage_idx]
                large_kernel_padding = large_kernel_paddings[stage_idx]
                stage = Sequential(*[
                    ConvNeXtBlockLarge(
                        in_channels=channels,
                        kernel_size=large_kernel_size,
                        padding=large_kernel_padding,
                        drop_path_rate=dpr[block_idx + j],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        linear_pw_conv=linear_pw_conv,
                        layer_scale_init_value=layer_scale_init_value)
                    for j in range(depth)
                ])
            else:
                stage = Sequential(*[
                    ConvNeXtBlock(
                        in_channels=channels,
                        drop_path_rate=dpr[block_idx + j],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        linear_pw_conv=linear_pw_conv,
                        layer_scale_init_value=layer_scale_init_value)
                    for j in range(depth)
                ])

            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        # x.shape [4, 96, 1024, 1024]
        outs = []
        for i, stage in enumerate(self.stages):
            if i >= self.first_downsample:
                x = self.downsample_layers[i](x)  # NOTE, pretrain_weight
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()
