import os
from typing import Any, Callable, Optional, List

import albumentations as A
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments

from utils.transforms import ResizePad


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


class ClassificationDataModule(LightningDataModule):

    @validate_arguments
    def __init__(
        self,
        data_dir: str,
        train_transforms: list,
        val_transforms: list,
        batch_size: int,
        val_split: float,
        classes: List[str] = None,
        input_shape: List[int] = None,
        test_split: float = 0.0,
        num_workers: int = os.cpu_count(),
        test_data_dir: str = None,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        use_weighted_sampler: bool = True,
    ) -> None:
        """
        :param data_dir: path to folder with data
        :param input_shape: list with model input image resolution
        :param classes: list with class names
        :param val_split: proportion of validation test
        :param test_split: proportion of test set (if test_data_dir is None)
        :param batch_size: the batch size
        :param num_workers: how many workers to use for loading data
        :param test_data_dir: path to folder with test data
        :param shuffle: if true shuffles the data every epoch
        :param pin_memory: if true, the data loader will copy Tensors into CUDA pinned memory
            before returning them
        :param drop_last: If true drops the last incomplete batch
        :param use_weighted_sampler: if true, use WeightedRandomSampler for train dataset
        """

        super().__init__()
        self.data_dir = data_dir
        self.train_tf_list = train_transforms
        self.val_tf_list = val_transforms
        self.input_shape = tuple(input_shape) if input_shape else input_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
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

        class_distribution = np.unique(
            train_dataset.targets, return_counts=True)[1]
        class_weights = 1.0 / class_distribution
        self.class_weights = torch.Tensor(class_weights / class_weights.max())

        val_len = round(val_split * len(train_dataset))
        test_len = round(test_split * len(train_dataset))
        train_len = len(train_dataset) - val_len - test_len

        idx = list(range(len(train_dataset)))
        if test_data_dir is None:
            # Create test dataset from train data
            if test_split == 1.0:
                # Create only test dataset
                train_idx = []
                val_idx = []
                test_idx = idx
            else:
                train_idx, val_idx = \
                    train_test_split(
                        idx,
                        train_size=train_len,
                        test_size=test_len+val_len,
                        stratify=train_dataset.targets,
                    )
                test_idx = []
                if test_split > 0:
                    val_idx, test_idx = \
                        train_test_split(
                            val_idx,
                            train_size=val_len,
                            test_size=test_len,
                            stratify=[train_dataset.targets[i] for i in val_idx],
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
            self.test_set = SubsetWithTargets(
                test_dataset,
                indices=test_idx,
                labels=[test_dataset.targets[i] for i in test_idx]
            )
        else:
            # Test dataset is in separate folder
            train_idx, val_idx = \
                train_test_split(
                    idx,
                    train_size=train_len,
                    test_size=test_len+val_len,
                    stratify=train_dataset.targets,
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
            self.test_set = test_dataset

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
                shuffle=self.shuffle,
            )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
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
        return lambda x: A.Compose(
                self.train_tf_list
            )(image=(np.array(x)/255.0).astype(np.float32))['image'].float()

    def val_transforms(self) -> Callable:
        return lambda x: A.Compose(
                self.val_tf_list
            )(image=(np.array(x)/255.0).astype(np.float32))['image'].float()
