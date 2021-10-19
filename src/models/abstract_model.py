from typing import OrderedDict

from torch import Tensor
from torch.nn import Module


class AbstractModel(Module):
    def forward(self, x):
        pass

    def get_model(self):
        return self

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        return self.get_model().load_state_dict(state_dict, strict=strict)