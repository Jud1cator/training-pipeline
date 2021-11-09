from effdet import create_model

from models import AbstractModel


class RossMobileDetV2(AbstractModel):
    def __init__(
            self,
            num_classes: int,
            pretrained: bool = False
    ):
        super().__init__()
        self.model = create_model(
            'mobiledetv2_110d',
            num_classes=num_classes+1,
            pretrained=pretrained
        )

    def forward(self, x, *args, **kwargs):
        return self.model.forward(x)

    def get_model(self, *args, **kwargs):
        return self.model
