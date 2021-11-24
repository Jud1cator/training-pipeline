import os
import warnings

from typing import Optional, Union, List
from pathlib import Path

import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, Dataset, DataLoader, Subset
from torchvision.datasets import CocoDetection

from utils.dataset import get_annotations_info


class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        super().__init__()
        self.coco_dataset = CocoDetection(
            root=img_dir,
            annFile=ann_file
        )
        self.transform = transform
        if self.transform:
            print('Filtering valid images...')
            valid_ids = []
            print(f'0/{len(self.coco_dataset.ids)}')
            for i in range(len(self.coco_dataset)):
                if i % 100 == 0:
                    print("\033[A\033[A")
                    print(f'{i}/{len(self.coco_dataset.ids)}')

                image, bboxes, labels = self.__getitem__(i)
                if len(bboxes) > 0:
                    valid_ids.append(i)
            print("\033[A\033[A")
            print(f'Kept {len(valid_ids)}/{len(self.coco_dataset)} images')
            self.coco_dataset = Subset(self.coco_dataset, valid_ids)

    def __len__(self):
        return len(self.coco_dataset)

    @staticmethod
    def apply_transforms(transform, image, bboxes, labels):
        tf = transform(image=image, bboxes=bboxes, labels=labels)
        image = torch.tensor(np.float32(tf['image']) / 255)
        bboxes = torch.tensor(
            np.array(tf['bboxes'], dtype=np.float32).reshape((len(tf['bboxes']), 4)),
            dtype=torch.float32
        )
        labels = torch.tensor(tf['labels'], dtype=torch.float32)
        return image, bboxes, labels

    def __getitem__(self, index):
        image, ann = self.coco_dataset[index]
        image = np.array(image, dtype=np.uint8)
        bboxes = [a['bbox'] for a in ann]
        labels = [a['category_id'] for a in ann]

        if self.transform:
            image, bboxes, labels = self.apply_transforms(self.transform, image, bboxes, labels)

        return image, bboxes, labels


class CocoDetectionSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        if self.transform:
            print('Filtering valid images...')
            valid_ids = []
            print(f'0/{len(self.subset)}')
            for i in range(len(self.subset)):
                if i % 100 == 0:
                    print("\033[A\033[A")
                    print(f'{i}/{len(self.subset)}')

                image, bboxes, labels = self.__getitem__(i)
                if len(bboxes) > 0:
                    valid_ids.append(i)
            print("\033[A\033[A")
            print(f'Kept {len(valid_ids)}/{len(self.subset)} images')
            self.subset = Subset(self.subset, valid_ids)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, bboxes, labels = self.subset[index]
        image = np.array(image, dtype=np.uint8)

        if self.transform:
            image, bboxes, labels = CocoDetectionDataset.apply_transforms(
                self.transform, image, bboxes, labels
            )

        for i in range(len(bboxes)):
            x, y, h, w = bboxes[i]
            bboxes[i] = torch.FloatTensor([y, x, y + h, x + w])

        return image, bboxes, labels


class DetectionDataModule(LightningDataModule):

    def __init__(
            self,
            images_dir: str,
            annotation_files: Union[List[str], str],
            image_size: List[int],
            batch_size: int,
            val_split: float = 0.0,
            test_split: float = 0.0,
            num_workers: int = os.cpu_count(),
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args,
            **kwargs
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
        super(DetectionDataModule, self).__init__(*args, **kwargs)
        self.images_dir = Path(images_dir)
        self.annotations = annotation_files
        self.image_size = tuple(image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        assert val_split + test_split < 1, f"Provided ratios for val and test splits \
            should not be greater than 1.0, got {val_split} for val \
            and {test_split} for test"

        self.parse_annotations(val_split, test_split)

    def parse_annotations(self, val_ratio: float, test_ratio: float):
        """Parse provided annotations.
            If there are several annotations files tries to parse them
            by the type (train, val, test).
            Otherwise, parse only one file and split it by provided
            ratios for sets.
        """
        if type(self.annotations) is list:
            # parse filenames to get internal info
            info, parsed = get_annotations_info(self.annotations)

            if info['val']:
                warnings.warn(
                    '[!] val_split will not be considered, '
                    'because validation set is already provided'
                )
                self.val_set = CocoDetectionDataset(
                    self.images_dir, ann_file=parsed['val_fn'], transform=self.val_transforms()
                )

            if info['test']:
                warnings.warn(
                    '[!] test_split will not be considered, because test set is already provided'
                )
                self.test_set = CocoDetectionDataset(
                    self.images_dir, ann_file=parsed['test_fn'], transform=self.val_transforms()
                )

            if info['unknown']:
                warnings.warn(
                    'Unknown annotations are found. '
                    'This annotations will be considered as train set'
                )
                self.train_set = CocoDetectionDataset(
                    self.images_dir, ann_file=parsed['unknown'], transform=self.train_transforms()
                )
            else:
                self.train_set = CocoDetectionDataset(
                    self.images_dir, ann_file=parsed['train_fn'], transform=self.train_transforms()
                )

        elif type(self.annotations) is str:
            # load annotation file and split it on train, val, test
            dataset = CocoDetectionDataset(self.images_dir, ann_file=self.annotations)
            total_len = len(dataset)

            # compute lengths for sets based on ratios
            val_len = int(total_len * val_ratio)
            test_len = int(total_len * test_ratio)
            train_len = total_len - val_len - test_len

            lengths = [train_len, val_len, test_len]

            train_data, val_data, test_data = random_split(dataset, lengths)

            # generate datasets out of subsets with denoted transforms
            self.train_set = CocoDetectionSubset(train_data, transform=self.train_transforms())
            self.val_set = CocoDetectionSubset(val_data, transform=self.val_transforms())
            # self.test_set = CocoDetectionSubset(test_data, transform=self.val_transforms())
        else:
            raise TypeError(f"Unacceptable type of annotations: {type(self.annotations)}")

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
            shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
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
            [
                A.CenterCrop(height=480, width=480),
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format='coco', min_area=0, min_visibility=0, label_fields=['labels']
            )
        )
        return transform

    def val_transforms(self):
        transform = A.Compose(
            [
                A.CenterCrop(height=480, width=480),
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format='coco', min_area=0, min_visibility=0, label_fields=['labels']
            )
        )
        return transform

    @staticmethod
    def collate_fn(batch):
        images, boxes, labels = [], [], []
        for item in batch:
            images.append(item[0])
            boxes.append(item[1])
            labels.append(item[2])
        images = torch.stack(images)

        return images, boxes, labels
