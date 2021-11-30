from geffnet import (
    efficientnet_lite0,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7
)

from models import AbstractModelWrapper


class EfficientNetLite0(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_lite0(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetB0(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b0(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetB1(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b1(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetB2(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b2(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetB3(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b3(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetB4(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b4(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetB5(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b5(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetB6(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b6(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model

class EfficientNetB7(AbstractModelWrapper):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b7(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model
