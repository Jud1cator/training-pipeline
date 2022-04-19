import unittest

import torch

from src.models.detection.utils.mobiledet_feature_extractor import (
    FeatureExtractor,
    FeatureMapBlock,
    MobileDetCPUBackbone
)


class TestSSDMobileDet(unittest.TestCase):

    def test_mobiledet_cpu_backbone(self):
        model = MobileDetCPUBackbone()
        r = model.forward(torch.rand((4, 3, 320, 320)))
        correct_values = [
            [4, 8, 160, 160],
            [4, 16, 80, 80],
            [4, 32, 40, 40],
            [4, 72, 20, 20],
            [4, 144, 10, 10]
        ]
        for i in range(len(correct_values)):
            self.assertListEqual(correct_values[i], list(r[i].shape))

    def test_feature_map_block(self):
        model = FeatureMapBlock()
        r = model.forward(torch.rand((4, 144, 10, 10)))
        self.assertListEqual([4, 512, 5, 5], list(r.shape))

    def test_feature_extractor(self):
        correct_values = [
            [4, 72, 20, 20],
            [4, 144, 10, 10],
            [4, 512, 5, 5],
            [4, 256, 3, 3],
            [4, 256, 2, 2],
            [4, 128, 1, 1]
        ]
        model = FeatureExtractor()
        r = model.forward(torch.rand((4, 3, 320, 320)))
        for i in range(len(correct_values)):
            self.assertListEqual(correct_values[i], list(r[i].shape))

    # def test_mobiledet_cpu(self):
    #     model = SSDMobileDetCPU(
    #         num_classes=1,
    #         anchor_params={
    #             'image_size': [320, 320],
    #             'min_level': 4,
    #             'max_level': 9,
    #             'num_scales': 1,
    #             'anchor_scale': 1
    #         }
    #     )
    #
    #     x_class, x_box = model.infer(torch.rand((4, 3, 320, 320)))
    #
    #     correct_values = [  # TODO: check why size differ with tensorflow model
    #         1200,
    #         300,  # 600 Original size
    #         75,   # 150
    #         27,   # 54
    #         12,   # 24
    #         3     # 6
    #     ]
    #     for i in range(len(correct_values)):
    #         batch_size, num_outputs, boxes_size = list(x_class[i].shape)
    #         self.assertEqual(num_outputs, model.num_classes)
    #         self.assertEqual(boxes_size, correct_values[i])
    #         self.assertEqual(batch_size, 4)
    #
    #         batch_size, num_outputs, boxes_size = list(x_box[i].shape)
    #         self.assertEqual(num_outputs, 4)
    #         self.assertEqual(boxes_size, correct_values[i])
    #         self.assertEqual(batch_size, 4)


if __name__ == '__main__':
    unittest.main()
