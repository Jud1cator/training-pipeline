from typing import Tuple

import torch.nn as nn
import torchvision
from timm.models.efficientnet_blocks import DepthwiseSeparableConv
from timm.models.layers import create_conv2d
from torch.nn import functional as F


class MobileNetV2Backbone(torchvision.models.MobileNetV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.classifier
        self.out_channels = (32, 96, 320)

    def get_stages(self):
        return [
            self.features[:7],   # 1x32x40x40
            self.features[7:14],  # 1x96x20x20
            self.features[14:18],  # 1x320x10x10
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('classifier.1.bias', None)
        state_dict.pop('classifier.1.weight', None)
        super().load_state_dict(state_dict, **kwargs)


class MNASFPNCell(nn.Module):
    def __init__(self, out_channels=48, in_channels=(48, 48, 48)):
        super(MNASFPNCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block0 = MNASFPNCellBlock(
            in_channels=(in_channels[1], in_channels[2]),
            expansion_size=256,
            out_channels=out_channels,
            upsample_scale=2,
            downsample_scale=1,
            sep_conv_kernel_size=1
        )

        self.block1 = MNASFPNCellBlock(
            in_channels=(in_channels[0], out_channels),
            expansion_size=128,
            out_channels=out_channels,
            upsample_scale=2,
            downsample_scale=1,
            sep_conv_kernel_size=1
        )

        self.block2 = MNASFPNCellBlock(
            in_channels=(out_channels, out_channels),
            expansion_size=128,
            out_channels=out_channels,
            upsample_scale=1,
            downsample_scale=2,
            sep_conv_kernel_size=1
        )

        self.block3 = MNASFPNCellBlock(
            in_channels=(out_channels, out_channels),
            expansion_size=128,
            out_channels=out_channels,
            upsample_scale=2,
            downsample_scale=2,
            sep_conv_kernel_size=1
        )

        self.block4 = MNASFPNCellBlock(
            in_channels=(out_channels, out_channels),
            expansion_size=96,
            out_channels=out_channels,
            upsample_scale=1,
            downsample_scale=4,
            sep_conv_kernel_size=1
        )

    def forward(self, x):
        block0_out = self.block0([x[1], x[2]])  # 1x48x20x20
        block1_out = self.block1([x[0], block0_out])  # 1x48x40x40
        block2_out = self.block2([block1_out, block0_out])  # 1x48x20x20
        block3_out = self.block3([block0_out, x[3]])  # 1x48x10x10
        block4_out = self.block4([block0_out, x[3]]) + x[3]  # 1x48x5x5

        if self.in_channels[0] == self.out_channels:
            block1_out += x[0]
        if self.in_channels[1] == self.out_channels:
            block2_out += x[1]
        if self.in_channels[2] == self.out_channels:
            block3_out += x[2]

        return [block1_out, block2_out, block3_out, block4_out]


class MNASFPNCellBlock(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, int],
            expansion_size: int = 256,
            out_channels: int = 48,
            upsample_scale: int = 2,
            downsample_scale: int = 1,
            sep_conv_kernel_size: int = 3
    ):
        super(MNASFPNCellBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='nearest') \
            if upsample_scale != 1 else None
        self.downsample = nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=downsample_scale,
            ceil_mode=True
        ) if downsample_scale != 1 else None
        self.conv_input0 = create_conv2d(in_channels[0], expansion_size, kernel_size=[1, 1])
        self.conv_input1 = create_conv2d(in_channels[1], expansion_size, kernel_size=[1, 1])
        self.sep_conv = DepthwiseSeparableConv(
            expansion_size, out_channels, dw_kernel_size=sep_conv_kernel_size
        )

    def forward(self, x):
        x0 = self.conv_input0(x[0])
        x1 = self.conv_input1(x[1])
        if self.downsample:
            x0 = self.downsample(x0)
        if self.upsample:
            x1 = self.upsample(x1)
        x = F.relu6(x0 + x1, inplace=True)
        x = self.sep_conv(x)
        return x


class MNASFPN(nn.Module):
    def __init__(
            self,
            out_channels: int = 48,
            backbone_out_channels: Tuple[int, int, int] = (32, 96, 320),
    ):
        super(MNASFPN, self).__init__()
        self.C6_downsample = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], ceil_mode=True)
        self.C6_conv = create_conv2d(backbone_out_channels[2], out_channels, kernel_size=1)
        self.cells = nn.ModuleList()
        self.cells.append(MNASFPNCell(out_channels=out_channels, in_channels=backbone_out_channels))
        in_channels_default = (out_channels, out_channels, out_channels)
        for i in range(3):
            self.cells.append(
                MNASFPNCell(out_channels=out_channels, in_channels=in_channels_default)
            )

    def forward(self, x):
        c3, c4, c5 = x
        c6 = self.C6_conv(self.C6_downsample(x[-1]))
        x = [c3, c4, c5, c6]
        for cell in self.cells:
            x = cell(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            out_channels: int = 48
    ):
        super(FeatureExtractor, self).__init__()
        self.backbone = MobileNetV2Backbone()
        self.mnasfpn = MNASFPN(
            out_channels=out_channels, backbone_out_channels=self.backbone.out_channels
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.mnasfpn(x)
        return x
