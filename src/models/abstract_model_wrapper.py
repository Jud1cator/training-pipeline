from typing import OrderedDict

from torch import Tensor
from torch.nn import Module


class AbstractModelWrapper(Module):
    """ Wrapper class for models used in training procedure """
    def forward(self, *args, **kwargs):
        self.get_model().forward(*args, **kwargs)

    def get_model(self, *args, **kwargs):
        return self

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        return self.get_model().load_state_dict(state_dict, strict=strict)
