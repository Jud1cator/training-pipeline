from typing import List, Tuple

import torch
from effdet import create_model

from src.models.detection.utils.detection_model_wrapper import DetectionModelWrapper


class EfficientDet(DetectionModelWrapper):
    def __init__(
            self,
            model_name,
            num_classes: int,
            anchor_params: dict,
            max_detection_points: int = 5000,
            soft_nms: bool = False,
            max_det_per_image: int = 100,
            pretrained: bool = False,
            pretrained_backbone: bool = False
    ):
        super().__init__(
            num_classes=num_classes,
            anchor_params=anchor_params,
            max_detection_points=max_detection_points,
            soft_nms=soft_nms,
            max_det_per_image=max_det_per_image
        )
        self.model = create_model(
            model_name,
            num_classes=self.num_classes,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone
        )

    def get_model(self):
        return self.model

    def infer(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.model.forward(torch.stack(x))
