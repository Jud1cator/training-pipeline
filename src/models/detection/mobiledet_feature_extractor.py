import functools
from typing import Tuple, Dict

import torch
from timm.models.efficientnet_blocks import SqueezeExcite, InvertedResidual
from timm.models.layers import create_conv2d, HardSwish
from timm.models.layers import hard_swish
from torch import nn, Tensor

from models.blocks import InvertedResidualNoExpansion


def _scale_filters(filters, multiplier, base=8):
    """Scale the filters accordingly to (multiplier, base)."""
    round_half_up = int(int(filters) * multiplier / base + 0.5)
    result = int(round_half_up * base)
    return max(result, base)


class MobileDetCPUBackbone(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            multiplier: float = 1.0,
    ):
        super(MobileDetCPUBackbone, self).__init__()

        def _scale(filters):
            return _scale_filters(filters, multiplier)

        ibn_creator = functools.partial(
            InvertedResidual, se_layer=SqueezeExcite, act_layer=HardSwish)

        self.conv1 = create_conv2d(in_channels, _scale(16), kernel_size=3, stride=2)
        self.ibn_no_exp_ratio1 = InvertedResidualNoExpansion(
            _scale(16), _scale(8), se_layer=SqueezeExcite, act_layer=HardSwish)

        ibn_layers = [
            ibn_creator(_scale(8), _scale(16), exp_ratio=4, stride=2, noskip=True),
            ibn_creator(_scale(16), _scale(32), exp_ratio=8, stride=2, noskip=True),
            ibn_creator(_scale(32), _scale(32), exp_ratio=4),
            ibn_creator(_scale(32), _scale(32), exp_ratio=4),
            ibn_creator(_scale(32), _scale(32), exp_ratio=4),
            ibn_creator(_scale(32), _scale(72), dw_kernel_size=5, exp_ratio=8, stride=2, noskip=True),
            ibn_creator(_scale(72), _scale(72), exp_ratio=8),
            ibn_creator(_scale(72), _scale(72), dw_kernel_size=5, exp_ratio=4),
            ibn_creator(_scale(72), _scale(72), exp_ratio=4),
            ibn_creator(_scale(72), _scale(72), exp_ratio=8, noskip=True),
            ibn_creator(_scale(72), _scale(72), exp_ratio=8),
            ibn_creator(_scale(72), _scale(72), exp_ratio=8),
            ibn_creator(_scale(72), _scale(72), exp_ratio=8),
            ibn_creator(_scale(72), _scale(104), dw_kernel_size=5, exp_ratio=8, stride=2, noskip=True),
            ibn_creator(_scale(104), _scale(104), dw_kernel_size=5, exp_ratio=4),
            ibn_creator(_scale(104), _scale(104), dw_kernel_size=5, exp_ratio=4),
            ibn_creator(_scale(104), _scale(104), exp_ratio=4),
            ibn_creator(_scale(104), _scale(144), exp_ratio=8, noskip=True)
        ]
        self.ibn_layers = nn.ModuleList(ibn_layers)
        self.out_channels = 144

    def forward(self, x):
        endpoints = {}
        x = hard_swish(self.conv1(x))
        x = self.ibn_no_exp_ratio1(x)
        endpoints['C1'] = x
        x = self.ibn_layers[0](x)
        endpoints['C2'] = x
        for i in range(1, 5):
            x = self.ibn_layers[i](x)
        endpoints['C3'] = x
        for i in range(5, 13):
            x = self.ibn_layers[i](x)
        endpoints['C4'] = x
        for i in range(13, 18):
            x = self.ibn_layers[i](x)
        endpoints['C5'] = x
        return endpoints


class FeatureMapBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 144,
            out_channels: int = 512,
            act_layer=nn.ReLU6,
            norm_layer=nn.BatchNorm2d,
    ):
        super(FeatureMapBlock, self).__init__()
        mid_channels = out_channels//2

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = norm_layer(mid_channels)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_channels, mid_channels, [3, 3], stride=2, depthwise=True)
        self.bn2 = norm_layer(mid_channels)
        self.act2 = act_layer(inplace=True)

        # Point-wise expansion
        self.conv_pw2 = create_conv2d(mid_channels, out_channels, [1, 1])
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

    def forward(self, x):
        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Point-wise expansion
        x = self.conv_pw2(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x


class MultiResolutionFeatureMaps(nn.Module):
    def __init__(
            self,
            in_channels: int = 144,
            blocks_out_channels: Tuple[int] = (512, 256, 256, 128),
    ):
        super(MultiResolutionFeatureMaps, self).__init__()
        self.blocks = []
        for out_channels in blocks_out_channels:
            self.blocks.append(FeatureMapBlock(in_channels=in_channels, out_channels=out_channels))
            in_channels = out_channels

    def forward(self, input: Dict[str, Tensor]):
        endpoints = {
            "BoxPredictor_0": input['C4'],
            "BoxPredictor_1": input['C5']
        }
        x = input['C5']
        for k, block in enumerate(self.blocks, 2):
            endpoints["BoxPredictor_{}".format(k)] = x = block(x)
        return endpoints


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            feature_maps_out_channels: Tuple[int] = (512, 256, 256, 128),
    ):
        super(FeatureExtractor, self).__init__()
        self.backbone = MobileDetCPUBackbone(in_channels=in_channels)
        self.multi_resolution_feature_map = MultiResolutionFeatureMaps(
            in_channels=self.backbone.out_channels,
            blocks_out_channels=feature_maps_out_channels
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.multi_resolution_feature_map(x)
        return x


if __name__ == "__main__":
    model = MobileDetCPUBackbone()
    r = model.forward(torch.rand((32, 3, 320, 320)))
    for n, x in r.items():
        print(x.shape)
