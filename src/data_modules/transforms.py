from src.data_modules.utils.helpers import resize_pad


class ResizePad:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, sample, *args, **kwargs):
        image = resize_pad(sample, self.target_shape)
        return image
