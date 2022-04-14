import math

from timm.models.layers import create_conv2d
from torch import nn

from src.models.utils.blocks import SeparableConv2d


class ConvolutionalBoxHead(nn.Module):
    def __init__(
            self,
            num_outputs,
            in_channels,
            out_channels=None,
            num_predictions_per_location=3,
            kernel_size=None,
            box_encodings_clip_range=None,
    ):
        super(ConvolutionalBoxHead, self).__init__()
        self._num_outputs = num_outputs
        self._kernel_size = kernel_size or [1, 1]
        out_channels = out_channels or in_channels
        self._box_encodings_clip_range = box_encodings_clip_range

        self._separable_conv = SeparableConv2d(in_channels, out_channels)
        self._conv = create_conv2d(
            out_channels,
            num_predictions_per_location * num_outputs,
            self._kernel_size,
            bias=True
        )

    def forward(self, x):
        x = self._separable_conv(x)
        x = self._conv(x)
        return x


class PredictionTower(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PredictionTower, self).__init__()
        conv = [
            SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, act_layer=nn.ReLU6)
        ]
        for _ in range(3):
            conv.append(
                SeparableConv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, act_layer=nn.ReLU6
                )
            )
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class WeightSharedConvolutionBoxPredictor(nn.Module):
    def __init__(
            self,
            num_classes,
            in_channels,
            out_channels=64
    ):
        super(WeightSharedConvolutionBoxPredictor, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.prediction_tower = PredictionTower(in_channels=in_channels, out_channels=out_channels)

        self.class_predictor = ConvolutionalBoxHead(
            num_outputs=num_classes, in_channels=out_channels
        )

        for n, m in self.class_predictor.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    # Set bias values so that the sigmoid activation of them is around 0.01
                    m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))

        self.box_predictor = ConvolutionalBoxHead(num_outputs=4, in_channels=out_channels)

    def forward(self, x):
        x = self.prediction_tower(x)
        return self.class_predictor(x), self.box_predictor(x)
