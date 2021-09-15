import os
from typing import Any, Union, Callable, Optional
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms as transforms
from torchvision.datasets import CocoDetection


def collate_fn(batch):
    return tuple(zip(*batch))


class COCODataModule(LightningDataModule):

    def __init__(
        self,
        data_path: str,
        ann_path: str,
        val_split: float = 0.2,
        test_split: float = 0.0,
        num_workers: int = 4,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
        transforms: Union[Callable, None] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_path: path to source data
            ann_path: path to annotation file
            val_split: size of validation test (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before returning them
            drop_last: If true drops the last incomplete batch
            transform: A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.ToTensor
            target_transform: A function/transform that takes in the target and transforms it.
            transforms: A function/transform that takes input sample and its target as entry and returns a transformed version.
        """

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        dataset = CocoDetection(
            root=data_path,
            annFile=ann_path,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms
        )

        val_len = round(val_split * len(dataset))
        test_len = round(test_split * len(dataset))
        train_len = len(dataset) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
        return loader

    def _default_transforms(self) -> Callable:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform
