from typing import Optional, List, Dict

import torchvision
from torch import Tensor

from models import AbstractModel


class SSDLite(AbstractModel):
    def __init__(
            self,
            pretrained: bool = False,
            num_classes: int = 91,
            pretrained_backbone: bool = False,
            trainable_backbone_layers: Optional[int] = None
    ):
        """
        :param pretrained: If True, returns a model pre-trained on COCO train2017
        :param num_classes: number of output classes of the model (including the background)
        :param pretrained_backbone: If True, returns a model with backbone pre-trained on Imagenet
        :param trainable_backbone_layers: number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable.
        """
        super().__init__()
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=pretrained,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers
        )

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None):
        return self.model.forward(images=images, targets=targets)

    def get_model(self):
        return self.model


class SSD300(AbstractModel):
    def __init__(
            self,
            pretrained: bool = False,
            num_classes: int = 91,
            pretrained_backbone: bool = False,
            trainable_backbone_layers: Optional[int] = None
    ):
        """
        :param pretrained: If True, returns a model pre-trained on COCO train2017
        :param num_classes: number of output classes of the model (including the background)
        :param pretrained_backbone: If True, returns a model with backbone pre-trained on Imagenet
        :param trainable_backbone_layers: number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable.
        """
        super().__init__()
        self.model = torchvision.models.detection.ssd300_vgg16(
            pretrained=pretrained,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers
        )

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None):
        return self.model.forward(images=images, targets=targets)

    def get_model(self):
        return self.model
