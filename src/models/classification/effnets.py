import torch
from geffnet import efficientnet_b0, efficientnet_lite0

from src.models.utils.abstract_model_wrapper import AbstractModelWrapper


class EfficientNetB0(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True, from_checkpoint=False):
        super().__init__()
        self.model = efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
        if from_checkpoint:
            self.model.load_state_dict(torch.load(from_checkpoint))

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetLite0(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True, from_checkpoint=False):
        super().__init__()
        self.model = efficientnet_lite0(num_classes=num_classes, pretrained=pretrained)
        if from_checkpoint:
            self.model.load_state_dict(torch.load(from_checkpoint))

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model
