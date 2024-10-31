import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16
from torch import nn
from torch.nn import functional as F


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,  # ccp: 9
                 out_channels,  # ccp: 64
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):  # ccp max

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        # self.linear2 = nn.Linear(1, self.units, bias=False)

        assert mode in ['max', 'avg', 'maxavg']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)  # [N, 20, 10] --> [N, 20, 64]
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()  # --> [N, 20, 64]
        x = F.relu(x)

        # pooling
        if self.mode == 'max':
            if aligned_distance is not None:  # ccp: None
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # todo [N,1,C]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)
        elif self.mode == 'maxavg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_avg = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)
            x_max = (x_max + x_avg) / 2.

        if self.last_vfe:  # ccp:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class SEPFNLayer(nn.Module):
    """Pillar Feature Net Layer + SE Block
    """

    def __init__(self,
                 in_channels,  # ccp: 9
                 out_channels,  # ccp: 64
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):  # ccp max

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        self.channel_attention = ChannelAttention(in_channels=self.units)

        assert mode in ['max', 'avg', 'maxavg']
        self.mode = mode

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        # inputs: [N, 20, 10]
        x = self.linear(inputs)  # [N, 20, 10] --> [N, 20, 64]
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()  # --> [N, 20, 64]
        x = F.relu(x)

        # apply channel attention
        x = self.channel_attention(x)

        # pooling
        if self.mode == 'max':
            if aligned_distance is not None:  # ccp: None
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]  # todo [N,1,C]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)
        elif self.mode == 'maxavg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_avg = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)
            x_max = (x_max + x_avg) / 2.

        if self.last_vfe:  # ccp:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, _, channels = x.size()

        # Squeeze: global average pooling along the temporal dimension
        squeeze = self.avg_pool(x.permute(0, 2, 1)).view(N, channels)

        # Excitation: two fully connected layers with ReLU activation
        excitation = self.fc(squeeze).view(N, channels, 1)

        # Scale: element-wise multiplication with input
        scaled_feature = x * excitation.permute(0, 2, 1)

        return scaled_feature


class SEPFNLayerV2(SEPFNLayer):
    """Pillar Feature Net Layer + SE Block
    """

    def __init__(self,
                 in_channels,  # ccp: 9
                 out_channels,  # ccp: 64
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):  # ccp max

        super(SEPFNLayerV2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            last_layer=last_layer,
            mode=mode,
        )
        self.channel_attention = ChannelAttentionV2(in_channels=20)  # M


class ChannelAttentionV2(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, M, channels = x.size()

        # Squeeze: global average pooling along the temporal dimension
        squeeze = self.avg_pool(x).view(N, M)

        # Excitation: two fully connected layers with ReLU activation
        excitation = self.fc(squeeze).view(N, M, 1)

        # Scale: element-wise multiplication with input
        scaled_feature = x * excitation

        return scaled_feature
