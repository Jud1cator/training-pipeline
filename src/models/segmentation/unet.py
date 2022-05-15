from torch import nn as nn

from src.models.utils.abstract_model_wrapper import AbstractModelWrapper
from src.models.utils.blocks import DoubleConv, Down, Up


class UNet(AbstractModelWrapper):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        pretrained: bool = False
    ):
        """
        :param num_classes: Number of output classes required
        :param input_channels: Number of channels in input images, defaults to 3
        :param num_layers: Number of layers in each side of U-net, defaults to 5
        :param features_start: Number of features in first layer, defaults to 64
        :param bilinear: Whether to use bilinear interpolation
            or transposed convolutions for upsampling, defaults to False
        :param pretrained: If True, returns a model pre-trained, defaults to False
        """
        if num_layers < 1:
            raise ValueError(
                f'num_layers = {num_layers}, expected: num_layers > 0'
            )

        if pretrained:
            raise NotImplementedError(
                'Pretrained UNet is not implemented yet.'
            )

        super().__init__()

        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward path of network
        """
        xi = [self.layers[0](x)]

        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        return self.layers[-1](xi[-1])
