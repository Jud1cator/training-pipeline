import os
import warnings
from math import floor
from typing import Any, Optional

import albumentations as A
import numpy as np
import torch
from PIL import Image
from pydantic import validate_arguments
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.data_modules.utils.stratified_split_optimizer import StratifiedSplitOptimizer
from src.utils.helpers import parse_image_resolution_from_transforms
from src.utils.typings import float_in_range, int_in_range, int_non_negative


class SemSegDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        masks_path: str,
        value_to_class_map: dict,
        ignore_index: Optional[int_in_range(0, 255)] = None,
        transform=None
    ):
        """
        :param images_path: path to folder with images
        :param masks_path: path to folder with masks
        :param value_to_class_map: dictionary which maps masks pixel values to class names
        """
        self.value_to_class_map = value_to_class_map
        self.value_to_idx_map = {
            v: k for k, v in dict(enumerate(self.value_to_class_map.keys())).items()
        }
        self.n_classes = len(self.value_to_class_map)
        self.ignore_index = ignore_index
        self.transform = transform
        self.img_path = images_path
        self.mask_path = masks_path
        self.img_list = sorted(self.get_filenames(self.img_path))
        self.mask_list = sorted(self.get_filenames(self.mask_path))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.img_list[idx]), dtype=np.uint8)
        mask = np.asarray(Image.open(self.mask_list[idx]), dtype=np.uint8)
        mask = self.encode_segmap(mask)
        if self.transform:
            tf = self.transform(image=img, mask=mask)
            img = tf['image']
            mask = tf['mask']
        return img, mask.long()

    def encode_segmap(self, mask):
        for class_value in self.value_to_class_map.keys():
            mask[mask == class_value] = self.value_to_idx_map[class_value]
        if self.ignore_index is not None:
            mask[mask >= self.n_classes] = self.ignore_index
        return mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class SemanticSegmentationDataModule(LightningDataModule):

    @validate_arguments
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        value_to_class_map: dict,
        train_transforms: list,
        val_transforms: list,
        batch_size: int_non_negative,
        ignore_index: Optional[int_in_range(0, 255)] = None,
        test_images_path: Optional[str] = None,
        test_masks_path: Optional[str] = None,
        train_split: float_in_range(0.0, 1.0) = 0.0,
        val_split: float_in_range(0.0, 1.0) = 0.0,
        test_split: float_in_range(0.0, 1.0) = 0.0,
        use_stratified_split: bool = False,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        drop_last: bool = False,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to folder with dataset
            val_split: size of validation test (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super().__init__(*args, **kwargs)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.value_to_class_map = value_to_class_map
        self.train_transforms_list = train_transforms
        self.val_transforms_list = val_transforms
        self.batch_size = batch_size
        self.ignore_index = ignore_index
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.image_resolution = parse_image_resolution_from_transforms(train_transforms)

        self.train_set = None
        self.val_set = None
        self.test_set = None

        assert 0.0 <= train_split + val_split + test_split <= 1.0, \
            'Sum of split lengths should be in range (0, 1)'

        dataset = SemSegDataset(
            self.images_dir,
            self.masks_dir,
            self.value_to_class_map,
            ignore_index,
            self.train_transforms()
        )
        n_classes = dataset.n_classes

        train_length = floor(train_split * len(dataset))
        val_length = floor(val_split * len(dataset))
        test_length = 0

        if test_images_path is not None and test_masks_path is not None:
            self.test_set = SemSegDataset(
                test_images_path,
                test_masks_path,
                self.value_to_class_map,
                ignore_index,
                self.val_transforms()
            )
        else:
            test_length = floor(test_split * len(dataset))

        unused_length = len(dataset) - train_length - val_length - test_length

        if unused_length > 0:
            dataset, _ = random_split(
                dataset,
                lengths=[len(dataset) - unused_length, unused_length]
            )
        if train_length > 0 and val_length > 0:
            self.train_set, val_dataset = \
                random_split(
                    dataset,
                    lengths=[train_length, val_length + test_length]
                )
            val_dataset.transform = self.val_transforms()
            if self.test_set is None:
                if test_length > 0:
                    self.val_set, self.test_set = random_split(
                        val_dataset,
                        lengths=[val_length, test_length]
                    )
                else:
                    self.val_set = val_dataset
                    self.test_set = self.val_set
                    warnings.warn(
                        'Test split is set to 0, validation set will be used for testing'
                    )

        if use_stratified_split and train_split > 0.0 and val_split > 0.0:
            split_ratios = [train_split, val_split]
            indices = [self.train_set.indices, self.val_set.indices]
            if test_split > 0.0:
                split_ratios.append(test_split)
                indices.append(self.test_set.indices)
            optimizer = StratifiedSplitOptimizer(dataset, n_classes, split_ratios)
            new_indices = optimizer.find_approximately_optimal_split(indices)
            self.train_set = Subset(dataset, new_indices[0])
            self.val_set = Subset(dataset, new_indices[1])
            if test_split > 0.0:
                self.test_set = Subset(dataset, new_indices[2])

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def train_transforms(self):
        return A.Compose(self.train_transforms_list)

    def val_transforms(self):
        return A.Compose(self.val_transforms_list)

    def create_dummy_input(self, batch_size):
        return torch.randn(batch_size, 3, self.image_resolution.height, self.image_resolution.width)
