from typing import Union

from geffnet import efficientnet_b0, efficientnet_lite0

from models import MetaModel


class EfficientNetB0(MetaModel):
    def __init__(
            self,
            input_shape=(224, 224),
            num_classes=1000,
            pretrained=True,
            checkpoint_path=None,
            **kwargs
    ):
        super().__init__(input_shape, num_classes, **kwargs)
        self.model = efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def forward(self, x):
        return self.model.forward(x)

    def get_model(self):
        return self.model

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict, **kwargs)


class EfficientNetLite0(MetaModel):
    def __init__(
            self,
            input_shape=(224, 224),
            num_classes=1000,
            pretrained=True,
            checkpoint_path=None,
            **kwargs
    ):
        super().__init__(input_shape, num_classes, **kwargs)
        self.model = efficientnet_lite0(num_classes=num_classes, pretrained=pretrained)
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def forward(self, x):
        return self.model.forward(x)

    def get_model(self):
        return self.model

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict, **kwargs)
