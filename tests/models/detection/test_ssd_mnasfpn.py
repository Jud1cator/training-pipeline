import unittest

import torch

from src.models.detection.mobiledets import SSDMobileNetV2MNASFPN
from src.models.detection.utils.convolutional_box_predictor import (
    PredictionTower,
    WeightSharedConvolutionBoxPredictor
)
from src.models.detection.utils.mnasfpn_feature_extractor import (
    MNASFPN,
    FeatureExtractor,
    MNASFPNCell,
    MNASFPNCellBlock,
    MobileNetV2Backbone
)


class TestSSDMnasFPN(unittest.TestCase):

    def test_mobilenetv2_backbone(self):
        model = MobileNetV2Backbone()
        r = model.forward(torch.rand((4, 3, 320, 320)))
        correct_values = [
            [4, 32, 40, 40],
            [4, 96, 20, 20],
            [4, 320, 10, 10]
        ]
        for i in range(len(correct_values)):
            self.assertListEqual(correct_values[i], list(r[i].shape))

    def test_MNASFPN_cell_block(self):
        cell_block_configs = [
            {
                'in_channels': (96, 320),
                'expansion_size': 256,
                'out_channels': 48,
                'upsample_scale': 2,
                'downsample_scale': 1,
                'sep_conv_kernel_size': 3
            },
            {
                'in_channels': (32, 48),
                'expansion_size': 128,
                'out_channels': 48,
                'upsample_scale': 2,
                'downsample_scale': 1,
                'sep_conv_kernel_size': 3
            },
            {
                'in_channels': (48, 48),
                'expansion_size': 128,
                'out_channels': 48,
                'upsample_scale': 1,
                'downsample_scale': 2,
                'sep_conv_kernel_size': 3
            },
            {
                'in_channels': (48, 48),
                'expansion_size': 128,
                'out_channels': 48,
                'upsample_scale': 2,
                'downsample_scale': 2,
                'sep_conv_kernel_size': 5
            },
            {
                'in_channels': (48, 48),
                'expansion_size': 96,
                'out_channels': 48,
                'upsample_scale': 1,
                'downsample_scale': 4,
                'sep_conv_kernel_size': 3
            }
        ]
        correct_shapes = [
            [4, 48, 20, 20],
            [4, 48, 40, 40],
            [4, 48, 20, 20],
            [4, 48, 10, 10],
            [4, 48, 5, 5],
        ]
        input_shapes = [
            [(4, 96, 20, 20), (4, 320, 10, 10)],
            [(4, 32, 40, 40), (4, 48, 20, 20)],
            [(4, 48, 40, 40), (4, 48, 20, 20)],
            [(4, 48, 20, 20), (4, 48, 5, 5)],
            [(4, 48, 20, 20), (4, 48, 5, 5)]
        ]
        for config, correct_shape, input_shape in zip(
                cell_block_configs, correct_shapes, input_shapes
        ):
            model = MNASFPNCellBlock(**config)
            r = model.forward([torch.rand(input_shape[0]), torch.rand(input_shape[1])])
            self.assertListEqual(correct_shape, list(r.shape))

    def test_MnasFPN_cell(self):
        correct_shapes = [
            [4, 48, 40, 40],
            [4, 48, 20, 20],
            [4, 48, 10, 10],
            [4, 48, 5, 5]
        ]
        model = MNASFPNCell(in_channels=(32, 96, 320))
        c6_out = torch.rand([4, 48, 5, 5])
        x0 = torch.rand([4, 32, 40, 40])
        x1 = torch.rand([4, 96, 20, 20])
        x2 = torch.rand([4, 320, 10, 10])
        x = (x0, x1, x2, c6_out)
        result = model.forward(x)
        for correct_shape, r in zip(correct_shapes, result):
            self.assertListEqual(correct_shape, list(r.shape))

    def test_MNASFPN(self):
        correct_values = [
            [4, 48, 40, 40],
            [4, 48, 20, 20],
            [4, 48, 10, 10],
            [4, 48, 5, 5],
        ]
        model = MNASFPN()
        x = [
            torch.rand([4, 32, 40, 40]),
            torch.rand([4, 96, 20, 20]),
            torch.rand([4, 320, 10, 10])
        ]
        result = model.forward(x)
        for i in range(len(correct_values)):
            self.assertListEqual(correct_values[i], list(result[i].shape))

    def test_FeatureExtractor(self):
        model = FeatureExtractor()
        correct_values = [
            [4, 48, 40, 40],
            [4, 48, 20, 20],
            [4, 48, 10, 10],
            [4, 48, 5, 5],
        ]
        result = model.forward(torch.rand([4, 3, 320, 320]))
        for i in range(len(correct_values)):
            self.assertListEqual(correct_values[i], list(result[i].shape))

    def test_PredictionTower(self):
        model = PredictionTower(in_channels=48, out_channels=64)
        for input_size in [40, 20, 10, 5]:
            result = model.forward(torch.rand([4, 48, input_size, input_size]))
            self.assertListEqual([4, 64, input_size, input_size], list(result.shape))

    def test_WeightSharedConvolutionBoxPredictor(self):
        model = WeightSharedConvolutionBoxPredictor(1, in_channels=48)
        shapes = [
            [4, 48, 40, 40],
            [4, 48, 20, 20],
            [4, 48, 10, 10],
            [4, 48, 5, 5],
        ]
        for shape in shapes:
            class_out, box_out = model.forward(torch.rand(shape))
            output_shape = shape.copy()
            output_shape[1] = 3 * 1
            self.assertListEqual(output_shape, list(class_out.shape))
            output_shape[1] = 3 * 4
            self.assertListEqual(output_shape, list(box_out.shape))

    def test_ssd_mnasfpn(self):
        model = SSDMobileNetV2MNASFPN(
            num_classes=1,
            anchor_params={
                'image_size': (320, 320),
                'min_level': 3,
                'max_level': 6,
                'num_scales': 3,
                'anchor_scale': 1.0
            }
        )
        model.eval()
        r = model.forward(torch.rand((4, 3, 320, 320)))
        self.assertListEqual(list(r.shape), [4, 100, 6])


if __name__ == '__main__':
    unittest.main()
