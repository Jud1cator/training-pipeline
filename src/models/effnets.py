from typing import Union

import torch
from geffnet import efficientnet_b0, efficientnet_lite0

from models import MetaModel


class EfficientNetB0(MetaModel):
    def __init__(self, input_shape=(224, 224), num_classes=1000,
                 pretrained=True, **kwargs):
        super().__init__(input_shape, num_classes, **kwargs)
        self.model = efficientnet_b0(
            num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        self.model.forward(x)

    def get_model(self):
        return self.model


class EfficientNetLite0(MetaModel):
    def __init__(self,
                 input_shape=(224, 224),
                 num_classes=1000,
                 pretrained=True,
                 from_checkpoint: Union[str, None] = None,
                 **kwargs):
        super().__init__(input_shape, num_classes, **kwargs)
        self.model = efficientnet_lite0(
            num_classes=num_classes, pretrained=pretrained)

        self.from_checkpoint(from_checkpoint)

    def forward(self, x):
        self.model.forward(x)

    def get_model(self):
        return self.model

    def from_checkpoint(self, path: Union[str, None]):
        if path:
            model_checkpoint = torch.load(path)
            self.get_model().load_state_dict(model_checkpoint, strict=False)
