import math
from typing import List, Tuple

import torch

from src.models.detection.utils.convolutional_box_predictor import (
    ConvolutionalBoxHead,
    WeightSharedConvolutionBoxPredictor
)
from src.models.detection.utils.detection_model_wrapper import DetectionModelWrapper
from src.models.detection.utils.mnasfpn_feature_extractor import FeatureExtractor as MNASFPNFE
from src.models.detection.utils.mobiledet_feature_extractor import \
    FeatureExtractor as SSDMobileDetFE


class SSDMobileDetCPU(DetectionModelWrapper):

    def __init__(
            self,
            num_classes: int,
            anchor_params: dict,
            max_detection_points: int = 1000,
            soft_nms: bool = False,
            max_det_per_image: int = 100
    ):
        super(SSDMobileDetCPU, self).__init__(
            num_classes=num_classes,
            anchor_params=anchor_params,
            max_detection_points=max_detection_points,
            soft_nms=soft_nms,
            max_det_per_image=max_det_per_image
        )
        self.feature_extractor = SSDMobileDetFE()

        heads_in_channels = [72, 144, 512, 256, 256, 128]
        classification_heads = []
        box_heads = []
        for channels in heads_in_channels:
            classification_heads.append(
                ConvolutionalBoxHead(
                    num_outputs=self.num_classes,
                    in_channels=channels,
                )
            )
            box_heads.append(
                ConvolutionalBoxHead(
                    num_outputs=4,
                    in_channels=channels
                )
            )
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        for n, m in self.classification_heads.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None:
                    # Set bias values so that the sigmoid activation of them is around 0.01
                    m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
        self.box_heads = torch.nn.ModuleList(box_heads)

    def infer(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = torch.stack(x)
        x_list = self.feature_extractor(x)
        x_class = []
        x_box = []
        for i in range(len(x_list)):
            x_class.append(self.classification_heads[i](x_list[i]))
            x_box.append(self.box_heads[i](x_list[i]))
        return x_class, x_box


class SSDMobileNetV2MNASFPN(DetectionModelWrapper):

    def __init__(
            self,
            num_classes: int,
            anchor_params: dict,
            max_detection_points: int = 1000,
            soft_nms: bool = False,
            max_det_per_image: int = 100
    ):
        super(SSDMobileNetV2MNASFPN, self).__init__(
            num_classes=num_classes,
            anchor_params=anchor_params,
            max_detection_points=max_detection_points,
            soft_nms=soft_nms,
            max_det_per_image=max_det_per_image
        )
        heads_in_channels = 48
        self.feature_extractor = MNASFPNFE(heads_in_channels)

        self.num_classes = num_classes + 1
        self.prediction_head = WeightSharedConvolutionBoxPredictor(
            self.num_classes, in_channels=heads_in_channels
        )

    def infer(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x_list = self.feature_extractor(x)
        x_class = []
        x_box = []
        for i in range(len(x_list)):
            x_class_i, x_box_i = self.prediction_head(x_list[i])
            x_class.append(x_class_i)
            x_box.append(x_box_i)
        return x_class, x_box
