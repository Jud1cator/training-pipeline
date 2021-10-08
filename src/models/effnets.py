from geffnet import efficientnet_b0, efficientnet_lite0

from models import AbstractModel


class EfficientNetB0(AbstractModel):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_b0(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model


class EfficientNetLite0(AbstractModel):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.model = efficientnet_lite0(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x, *args, **kwargs):
        self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model
