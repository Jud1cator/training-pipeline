import os
import warnings
from typing import Callable, List, Optional

import albumentations as A
import numpy as np
import torch
from pydantic import validate_arguments
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data_modules.utils.helpers import SubsetWithTargets
from src.utils.helpers import parse_image_resolution_from_transforms
from src.utils.typings import float_in_range, int_non_negative


class ClassificationDataModule(LightningDataModule):

    @validate_arguments
    def __init__(
        self,
        data_dir: str,
        train_transforms: list,
        val_transforms: list,
        batch_size: int_non_negative,
        classes: Optional[List[str]] = None,
        train_split: float_in_range(0.0, 1.0) = 0.0,
        val_split: float_in_range(0.0, 1.0) = 0.0,
        test_data_dir: Optional[str] = None,
        test_split: float_in_range(0.0, 1.0) = 0.0,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        drop_last: bool = False,
        use_weighted_sampler: bool = False,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
    ):
        """
        :param data_dir: path to folder with data
        :param train_transforms: list of transforms for input images at training step
        :param val_transforms: list of transforms for input images at validation step
        :param batch_size: the batch size of DataLoader
        :param classes: list with class names to use for training. If None, all classes are used.
        :param train_split: fraction of data to use for training set
        :param val_split: fraction of data to use for validation set
        :param test_data_dir: path to folder with testing data
        :param test_split: fraction of data to use for testing set if separate test set is not
            provided
        :param shuffle_train: whether to shuffle train set each epoch
        :param shuffle_val: whether to shuffle validation set each epoch
        :param drop_last: If true drops the last incomplete batch
        :param use_weighted_sampler: if true, use WeightedRandomSampler to draw samples from minor
            classes more frequently
        :param num_workers: how many workers to use for data loading
        :param pin_memory: if true, the data loader will copy Tensors into CUDA pinned memory
            before returning them
        """

        super().__init__()
        self.data_dir = data_dir
        self.image_resolution = parse_image_resolution_from_transforms(train_transforms)
        self.train_transforms_list = train_transforms
        self.val_transforms_list = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_weighted_sampler = use_weighted_sampler

        train_dataset = ImageFolder(
            self.data_dir,
            transform=self.train_transforms()
        )
        val_dataset = ImageFolder(
            self.data_dir,
            transform=self.val_transforms()
        )
        test_dataset = ImageFolder(
            self.data_dir if test_data_dir is None else test_data_dir,
            transform=self.val_transforms()
        )

        self.classes = classes if classes else train_dataset.classes

        class_distribution = np.unique(train_dataset.targets, return_counts=True)[1]
        class_weights = 1.0 / class_distribution
        self.class_weights = torch.Tensor(class_weights / class_weights.max())

        assert 0.0 <= train_split + val_split + test_split <= 1.0, \
            'Sum of split lengths should be in range (0, 1)'
        val_len = round(val_split * len(train_dataset))
        test_len = round(test_split * len(train_dataset))
        train_len = round(train_split * len(train_dataset))
        unused_len = len(train_dataset) - train_len - val_len - test_len

        idx = list(range(len(train_dataset)))

        self.train_set = None
        self.val_set = None
        self.test_set = None

        # Prepare test set
        if test_data_dir is not None:
            self.test_set = test_dataset
        else:
            self.test_set = None
            if test_split > 0:
                idx, test_idx = train_test_split(
                    idx,
                    train_size=(train_len + val_len + unused_len),
                    test_size=test_len,
                    stratify=train_dataset.targets,
                )
                self.test_set = SubsetWithTargets(
                    test_dataset,
                    indices=test_idx,
                    labels=[test_dataset.targets[i] for i in test_idx]
                )

        # Prepare train and val splits
        if 0 < unused_len < len(train_dataset):
            idx, _ = \
                train_test_split(
                    idx,
                    train_size=(train_len + val_len),
                    test_size=unused_len,
                    stratify=[train_dataset.targets[i] for i in idx],
                )
        if train_len > 0 and val_len > 0:
            train_idx, val_idx = \
                train_test_split(
                    idx,
                    train_size=train_len,
                    test_size=val_len,
                    stratify=[train_dataset.targets[i] for i in idx]
                )
            self.train_set = SubsetWithTargets(
                train_dataset,
                indices=train_idx,
                labels=[train_dataset.targets[i] for i in train_idx]
            )
            self.val_set = SubsetWithTargets(
                val_dataset,
                indices=val_idx,
                labels=[val_dataset.targets[i] for i in val_idx]
            )
        else:
            warnings.warn(
                'Length of either train or val splits is 0. Training stage will be skipped.'
            )
        if self.test_set is None and self.val_set is not None:
            warnings.warn(
                'Length of test split is 0. Validation split will be used for testing'
            )
            self.test_set = self.val_set

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        if self.use_weighted_sampler:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=[self.class_weights[i] for i in self.train_set.targets],
                num_samples=len(self.train_set),
                replacement=True
            )
            loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                sampler=sampler,
                shuffle=False,
            )
        else:
            loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle_train,
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

    def test_dataloader(self):
        loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def train_transforms(self) -> Callable:
        return lambda x: A.Compose(self.train_transforms_list)(image=np.array(x))['image']

    def val_transforms(self) -> Callable:
        return lambda x: A.Compose(self.val_transforms_list)(image=np.array(x))['image']

    def create_dummy_input(self, batch_size):
        return torch.randn(batch_size, 3, self.image_resolution.height, self.image_resolution.width)
