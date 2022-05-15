from PIL import Image
from torch.utils.data import Dataset, Subset


class SubsetWithTargets(Dataset):
    """
    Subset of a dataset at specified indices.

    :param dataset: torch.utils.data.Dataset object
    :param indices: sequence of indexes to sample from the dataset
    :param labels: sequence of class labels of the dataset
    """

    def __init__(self, dataset, indices, labels):
        self.dataset = Subset(dataset, indices)
        self.targets = labels

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return image, target

    def __len__(self):
        return len(self.targets)


def resize_pad(img, target_shape):
    """Resize image to target shape preserving aspect ratio with padding"""
    out = Image.new('RGB', target_shape)
    resize_ratio = min(
        target_shape[0] / img.size[0], target_shape[1] / img.size[1])
    resize_shape = (
        int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
    out.paste(img.resize(resize_shape, Image.BILINEAR))
    return out


def calculate_dataset_distribution(imgs):
    """Calculates mean and var of images for normalization"""
    pass
