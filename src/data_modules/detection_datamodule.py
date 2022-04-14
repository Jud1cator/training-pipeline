import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import albumentations as A
import numpy as np
import torch
from pydantic import validate_arguments
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CocoDetection

from src.utils.helpers import parse_image_resolution_from_transforms
from src.utils.typings import float_in_range, int_non_negative


class CocoDetectionDatasetWrapper(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            bbox_format: str,
            bbox_to_yxyx: bool,
            transform: Optional[A.Compose] = None,
            filter_images: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.bbox_format = bbox_format
        self.transform = transform
        self.bbox_to_yxyx = bbox_to_yxyx
        dataset_len = len(self.dataset)
        if filter_images and self.transform:
            print('Filtering images...')
            valid_ids = []
            print(f'0/{dataset_len}')
            for i in range(dataset_len):
                if i % 100 == 0:
                    print('\033[A\033[A')
                    print(f'{i}/{dataset_len}')
                image, bboxes, labels = self.__getitem__(i)
                if len(bboxes) > 0:
                    valid_ids.append(i)
            print('\033[A\033[A')
            print(f'Kept {len(valid_ids)}/{dataset_len} images')
            self.dataset = Subset(self.dataset, valid_ids)

    def __len__(self):
        return len(self.dataset)

    def apply_transforms(self, transform, image, bboxes, labels):
        tf = transform(image=image, bboxes=bboxes, labels=labels)
        image = tf['image']
        bboxes = np.array(tf['bboxes'], dtype=np.float32).reshape((len(tf['bboxes']), 4))
        if self.bbox_format == 'coco':
            # x, y, w, h -> x, y, x, y
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        if self.bbox_to_yxyx:
            # x, y, x, y -> y, x, y, x
            bboxes = bboxes[:, [1, 0, 3, 2]]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(tf['labels'], dtype=torch.float32)
        return image, bboxes, labels

    def __getitem__(self, index):
        image, ann = self.dataset[index]
        image = np.array(image, dtype=np.uint8)
        bboxes = [a['bbox'] for a in ann]
        labels = [a['category_id'] for a in ann]
        if self.transform:
            image, bboxes, labels = self.apply_transforms(self.transform, image, bboxes, labels)
        target = {
            'bboxes': bboxes,
            'labels': labels,
            'image_id': torch.tensor([index]),
        }

        return image, target, index


class DetectionDataModule(LightningDataModule):

    BBOX_FORMATS = ['coco', 'pascal_voc']

    @validate_arguments
    def __init__(
        self,
        images_dir: str,
        annotation_files: Union[Dict[str, str], str],
        bbox_format: str,
        bbox_to_yxyx: bool,
        train_transforms: list,
        val_transforms: list,
        batch_size: int_non_negative,
        filter_empty_gt: bool = False,
        train_split: float_in_range(0.0, 1.0) = 0.0,
        val_split: float_in_range(0.0, 1.0) = 0.0,
        test_split: float_in_range(0.0, 1.0) = 0.0,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        drop_last: bool = False,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
    ) -> None:
        """
        Args:
            :param images_dir: path to folder with images
            :param annotation_files: path to annotation file(s)
            :param input_size: required image size after preprocessing
            :param batch_size: the batch size
            :param val_split: fraction of validation data
            :param test_split: fraction of test data
            :param num_workers: how many workers to use for loading data
            :param shuffle: if true shuffles the data every epoch
            :param pin_memory: if true, the data loader will copy Tensors
                into CUDA pinned memory before returning them
            :param drop_last: If true drops the last incomplete batch
        """
        super(DetectionDataModule, self).__init__()
        self.images_dir = Path(images_dir)
        self.image_resolution = parse_image_resolution_from_transforms(train_transforms)
        if bbox_format not in self.BBOX_FORMATS:
            raise ValueError(f'Unknown format of bounding boxes: {bbox_format}')
        self.bbox_format = bbox_format
        self.bbox_to_yxyx = bbox_to_yxyx
        self.train_transforms_list = train_transforms
        self.val_transforms_list = val_transforms
        self.batch_size = batch_size
        self.filter_empty_gt = filter_empty_gt
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_set = None
        self.val_set = None
        self.test_set = None

        assert 0.0 <= train_split + val_split + test_split <= 1.0, \
            'Sum of split lengths should be in range (0, 1)'

        self.parse_annotations(annotation_files, train_split, val_split, test_split)

    def parse_annotations(
            self,
            annotation_files: Union[Dict[str, str], str],
            train_ratio: float_in_range(0.0, 1.0),
            val_ratio: float_in_range(0.0, 1.0),
            test_ratio: float_in_range(0.0, 1.0)
    ):
        """
        Parse provided annotations.
        If there are several annotations files tries to parse them
        by the type (train, val, test).
        Otherwise, parse only one file and split it by provided
        ratios for sets.
        """
        if type(annotation_files) is dict:
            # parse filenames to get internal info
            if 'train' not in annotation_files:
                warnings.warn(
                    'Training annotations are not found. Training stage will be skipped.'
                )
            elif 'val' in annotation_files:
                train_dataset = CocoDetection(
                    root=str(self.images_dir),
                    annFile=annotation_files['train']
                )
                self.train_set = CocoDetectionDatasetWrapper(
                    train_dataset,
                    self.bbox_format,
                    self.bbox_to_yxyx,
                    transform=self.train_transforms(),
                    filter_images=self.filter_empty_gt
                )
                val_dataset = CocoDetection(
                    root=str(self.images_dir),
                    annFile=annotation_files['val']
                )
                self.val_set = CocoDetectionDatasetWrapper(
                    val_dataset,
                    self.bbox_format,
                    self.bbox_to_yxyx,
                    transform=self.val_transforms(),
                    filter_images=self.filter_empty_gt
                )
            else:
                if val_ratio == 0.0:
                    raise RuntimeError(
                        'Validation annotations are not found and validation split is set to 0.'
                    )
                else:
                    dataset = CocoDetection(
                        root=str(self.images_dir),
                        annFile=annotation_files['train']
                    )
                    total_len = len(dataset)
                    val_len = int(total_len * val_ratio)
                    train_len = total_len - val_len
                    train_data, val_data = random_split(dataset, [train_len, val_len])
                    self.train_set = CocoDetectionDatasetWrapper(
                        train_data,
                        self.bbox_format,
                        self.bbox_to_yxyx,
                        transform=self.train_transforms(),
                        filter_images=self.filter_empty_gt
                    )
                    self.val_set = CocoDetectionDatasetWrapper(
                        val_data,
                        self.bbox_format,
                        self.bbox_to_yxyx,
                        transform=self.val_transforms(),
                        filter_images=self.filter_empty_gt
                    )
            if 'test' in annotation_files:
                test_dataset = CocoDetection(
                    root=str(self.images_dir),
                    annFile=annotation_files['test']
                )
                self.test_set = CocoDetectionDatasetWrapper(
                    test_dataset,
                    self.bbox_format,
                    self.bbox_to_yxyx,
                    transform=self.val_transforms(),
                    filter_images=self.filter_empty_gt
                )
            else:
                warnings.warn(
                    'Test annotations are not provided, validation set will be used for testing'
                )
                self.test_set = self.val_set

        elif type(annotation_files) is str:
            # load annotation file and split it on train, val, test
            dataset = CocoDetection(root=str(self.images_dir), annFile=annotation_files)
            total_len = len(dataset)

            # compute lengths for sets based on ratios
            val_len = int(total_len * val_ratio)
            test_len = int(total_len * test_ratio)
            train_len = int(total_len * train_ratio)

            lengths = [train_len, val_len, test_len]
            if sum(lengths) != total_len:
                lengths.append(total_len - sum(lengths))
                train_data, val_data, test_data, _ = random_split(dataset, lengths)
            else:
                train_data, val_data, test_data = random_split(dataset, lengths)

            # generate datasets out of subsets with denoted transforms
            if train_len > 0:
                self.train_set = CocoDetectionDatasetWrapper(
                    train_data,
                    self.bbox_format,
                    self.bbox_to_yxyx,
                    transform=self.train_transforms(),
                    filter_images=self.filter_empty_gt
                )
            if val_len > 0:
                self.val_set = CocoDetectionDatasetWrapper(
                    val_data,
                    self.bbox_format,
                    self.bbox_to_yxyx,
                    transform=self.val_transforms(),
                    filter_images=self.filter_empty_gt
                )
            if test_len > 0:
                self.test_set = CocoDetectionDatasetWrapper(
                    test_data,
                    self.bbox_format,
                    self.bbox_to_yxyx,
                    transform=self.val_transforms(),
                    filter_images=self.filter_empty_gt
                )
            else:
                warnings.warn(
                    'Test split is set to 0, validation set will be used for testing'
                )
                self.test_set = self.val_set
        else:
            raise TypeError(f'Unsupported type of annotations: {type(annotation_files)}')

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train,
            collate_fn=self.collate_fn
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        return loader

    def train_transforms(self):
        transform = A.Compose(
            self.train_transforms_list,
            bbox_params=A.BboxParams(
                format=self.bbox_format, min_area=0, min_visibility=0, label_fields=['labels']
            )
        )
        return transform

    def val_transforms(self):
        transform = A.Compose(
            self.val_transforms_list,
            bbox_params=A.BboxParams(
                format=self.bbox_format, min_area=0, min_visibility=0, label_fields=['labels']
            )
        )
        return transform

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def create_dummy_input(self, batch_size):
        return [torch.randn(
            3,
            self.image_resolution.height,
            self.image_resolution.width
        )] * batch_size
