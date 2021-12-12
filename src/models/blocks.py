import torch.nn as nn

from models.utils import conv_output_shape


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.pooling = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.pooling(x)
        return x

    def get_output_shape(self, input_shape):
        output_shape = conv_output_shape(
            input_shape,
            self.conv1.kernel_size,
            self.conv1.stride,
            self.conv1.padding
        )
        output_shape = conv_output_shape(
            output_shape,
            self.conv2.kernel_size,
            self.conv2.stride,
            self.conv2.padding
        )
        output_shape = (output_shape[0] // 2, output_shape[1] // 2)
        return output_shape
