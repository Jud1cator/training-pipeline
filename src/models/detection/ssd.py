from typing import Dict, List, Optional

import torchvision
from torch import Tensor

from src.models.utils.abstract_model_wrapper import AbstractModelWrapper


def coco_image_preprocessor(images: List[Tensor]):
    return [img / 255.0 for img in images]


def coco_targets_preprocessor(targets: List[Dict[str, Tensor]]):
    for i in range(len(targets)):
        targets[i]['boxes'][:, 2] = targets[i]['boxes'][:, 0] + targets[i]['boxes'][:, 2]
        targets[i]['boxes'][:, 3] = targets[i]['boxes'][:, 1] + targets[i]['boxes'][:, 3]
    return targets


class SSDLite(AbstractModelWrapper):
    def __init__(
            self,
            pretrained: bool = False,
            num_classes: int = 91,
            pretrained_backbone: bool = False,
            trainable_backbone_layers: Optional[int] = None,
            data_format: str = 'coco'
    ):
        """
        :param pretrained: If True, returns a model pre-trained on COCO train2017
        :param num_classes: number of output classes of the model (including the background)
        :param pretrained_backbone: If True, returns a model with backbone pre-trained on Imagenet
        :param trainable_backbone_layers: number of trainable (not frozen) resnet layers starting
            from final block. Valid values are between 0 and 6, with 6 meaning all backbone layers
            are trainable.
        """
        super().__init__()
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=pretrained,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers
        )
        if data_format == 'coco':
            self.images_preprocessor = coco_image_preprocessor
            self.targets_preprocessor = coco_targets_preprocessor
        else:
            self.images_preprocessor = lambda x: x
            self.targets_preprocessor = lambda x: x

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None):
        images = self.images_preprocessor(images)
        if targets is not None:
            targets = self.targets_preprocessor(targets)
        return self.model.forward(images=images, target=targets)

    def get_model(self):
        return self.model


class SSD300(AbstractModelWrapper):
    def __init__(
            self,
            pretrained: bool = False,
            num_classes: int = 91,
            pretrained_backbone: bool = False,
            trainable_backbone_layers: Optional[int] = None,
            data_format: str = 'coco'
    ):
        """
        :param pretrained: If True, returns a model pre-trained on COCO train2017
        :param num_classes: number of output classes of the model (including the background)
        :param pretrained_backbone: If True, returns a model with backbone pre-trained on Imagenet
        :param trainable_backbone_layers: number of trainable (not frozen) resnet layers starting
            from final block. Valid values are between 0 and 6, with 6 meaning all backbone layers
            are trainable.
        """
        super().__init__()
        self.model = torchvision.models.detection.ssd300_vgg16(
            pretrained=pretrained,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers
        )
        if data_format == 'coco':
            self.images_preprocessor = coco_image_preprocessor
            self.targets_preprocessor = coco_targets_preprocessor
        else:
            self.images_preprocessor = lambda x: x
            self.targets_preprocessor = lambda x: x

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None):
        images = self.images_preprocessor(images)
        if targets:
            targets = self.targets_preprocessor(targets)
        return self.model.forward(images=images, target=targets)

    def get_model(self):
        return self.model
