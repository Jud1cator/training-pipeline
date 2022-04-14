import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Swish, create_conv2d, drop_path
from timm.models.layers.helpers import make_divisible

from src.models.utils.helpers import conv_output_shape


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


class InvertedResidualNoExpansion(nn.Module):
    """ Inverted residual block w/ optional SE without the first 1x1 expansion convolution """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidualNoExpansion, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size,
                                      padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            info = dict(module='conv_pwl', hook_type='forward_pre',
                        num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding='',
            bias=False,
            channel_multiplier=1.0,
            pw_kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=Swish
    ):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias
        )

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bilinear: bool = False
    ):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        return self.conv(torch.cat([x2, x1], dim=1))
