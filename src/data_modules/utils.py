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


def calculate_dataset_distribution(imgs):
    """Calculates mean and var of images for normalization"""
    pass

