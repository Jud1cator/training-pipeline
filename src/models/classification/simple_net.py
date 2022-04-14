from typing import Tuple

from torch import nn

from src.models.utils.abstract_model_wrapper import AbstractModelWrapper
from src.models.utils.blocks import ConvBlock


class SimpleNet(AbstractModelWrapper):
    def __init__(self, input_resolution: Tuple[int, int], num_classes: int):
        super().__init__()
        input_resolution = tuple(input_resolution)
        self.conv1 = ConvBlock(3, 32)
        shape = self.conv1.get_output_shape(input_resolution)
        self.conv2 = ConvBlock(32, 64)
        shape = self.conv2.get_output_shape(shape)
        self.conv3 = ConvBlock(64, 128)
        shape = self.conv3.get_output_shape(shape)
        n_features = 128 * shape[0] * shape[1]
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.fc1 = nn.Linear(n_features, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        return x
