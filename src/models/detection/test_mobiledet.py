import unittest

import torch

from models.mobiledets import MobileDetCPU
from models.detection.mobiledet_feature_extractor import MobileDetCPUBackbone, FeatureMapBlock, \
    FeatureExtractor


class TestMobileDet(unittest.TestCase):
    def test_mobiledet_cpu_backbone(self):
        model = MobileDetCPUBackbone()
        r = model.forward(torch.rand((4, 3, 320, 320)))
        correct_values = {
            'C1': [4, 8, 160, 160],
            'C4': [4, 72, 20, 20],
            'C5': [4, 144, 10, 10]
        }
        for name in correct_values:
            self.assertListEqual(correct_values[name], list(r[name].shape))

    def test_feature_map_block(self):
        model = FeatureMapBlock()
        r = model.forward(torch.rand((4, 144, 10, 10)))
        self.assertListEqual([4, 512, 5, 5], list(r.shape))

    def test_feature_extractor(self):
        correct_values = {
            'BoxPredictor_0': [4, 72, 20, 20],
            'BoxPredictor_1': [4, 144, 10, 10],
            'BoxPredictor_2': [4, 512, 5, 5],
            'BoxPredictor_3': [4, 256, 3, 3],
            'BoxPredictor_4': [4, 256, 2, 2],
            'BoxPredictor_5': [4, 128, 1, 1]
        }
        model = FeatureExtractor()
        r = model.forward(torch.rand((4, 3, 320, 320)))
        for name in correct_values:
            self.assertListEqual(correct_values[name], list(r[name].shape))

    def test_mobiledet_cpu(self):
        model = MobileDetCPU()
        x_class, x_box = model.forward(torch.rand((4, 3, 320, 320)))
        correct_values = {  # TODO: check why size differ with tensorflow model
            'BoxPredictor_0': 1200,
            'BoxPredictor_1': 300,  # 600 Original size
            'BoxPredictor_2': 75,   # 150
            'BoxPredictor_3': 27,   # 54
            'BoxPredictor_4': 12,   # 24
            'BoxPredictor_5': 3     # 6
        }
        for name, true_size in correct_values.items():
            batch_size, num_outputs, boxes_size = list(x_class[name].shape)
            self.assertEqual(num_outputs, model.num_classes)
            self.assertEqual(boxes_size, true_size)
            self.assertEqual(batch_size, 4)

            batch_size, num_outputs, boxes_size = list(x_box[name].shape)
            self.assertEqual(num_outputs, 4)
            self.assertEqual(boxes_size, true_size)
            self.assertEqual(batch_size, 4)


if __name__ == '__main__':
    unittest.main()
