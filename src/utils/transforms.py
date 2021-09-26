from PIL import Image


def resize_pad(img, target_shape):
    """Resize image to target shape preserving aspect ratio with padding"""
    out = Image.new("RGB", target_shape)
    resize_ratio = min(
        target_shape[0] / img.size[0], target_shape[1] / img.size[1])
    resize_shape = (
        int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
    out.paste(img.resize(resize_shape, Image.BILINEAR))
    return out


class ResizePad:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, sample, *args, **kwargs):
        image = resize_pad(sample, self.target_shape)
        return image
