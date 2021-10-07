from typing import Tuple

from torch.nn import Module


class AbstractModel(Module):
    def __init__(
            self,
            input_shape: Tuple[int, int],
            num_classes: int,
            **kwargs
    ):
        super().__init__()

    def forward(self, x):
        pass

    def get_model(self):
        return self
