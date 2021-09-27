from typing import Union

import torch
from geffnet import efficientnet_b0, efficientnet_lite0

from models import MetaModel


class EfficientNetB0(MetaModel):
    def __init__(
            self,
            from_checkpoint: Union[str, None] = None,
            input_shape=(224, 224),
            num_classes=1000,
            pretrained=True,
            **kwargs
    ):
        super().__init__(input_shape, num_classes, **kwargs)
        self.model = efficientnet_b0(
            num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.model.forward(x)

    def get_model(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class EfficientNetLite0(MetaModel):
    def __init__(
            self,
            from_checkpoint: Union[str, None] = None,
            input_shape=(224, 224),
            num_classes=1000,
            pretrained=True,
            **kwargs
    ):
        super().__init__(input_shape, num_classes, **kwargs)
        self.model = efficientnet_lite0(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.model.forward(x)

    def get_model(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
