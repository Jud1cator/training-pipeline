from typing import Dict

from torch import Tensor

from effdet import create_model

from models import AbstractModelWrapper


class MobileDetV2Wrapper(AbstractModelWrapper):
    def __init__(
            self,
            num_classes: int,
            pretrained: bool = False
    ):
        super().__init__()
        # Add class for background
        num_classes += 1
        self.model = create_model(
            'mobiledetv2_110d',
            bench_task='train',
            bench_labeler=True,
            num_classes=num_classes,
            pretrained_backbone=pretrained
        )

    def forward(self, x, target: Dict[str, Tensor]):
        return self.model.forward(x, target)

    def get_model(self):
        return self.model
