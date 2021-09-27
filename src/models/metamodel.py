from typing import Tuple, Optional

import torch
from torch.nn import Module


class MetaModel(Module):
    def __init__(
            self,
            input_shape: Tuple[int, int],
            num_classes: int,
            checkpoint_path: Optional[str] = None,
            **kwargs
    ):
        super().__init__()

    def forward(self, x):
        pass

    def get_model(self):
        return self

    def load_from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)
