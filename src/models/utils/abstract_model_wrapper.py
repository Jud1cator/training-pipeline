from torch.nn import Module


class AbstractModelWrapper(Module):
    def forward(self, *args, **kwargs):
        self.get_model().forward(*args, **kwargs)

    def get_model(self, *args, **kwargs):
        return self
