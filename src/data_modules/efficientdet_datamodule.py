import warnings

import albumentations as A
import PIL
import numpy as np
import torch

from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from pytorch_lightning import LightningDataModule

from utils.visualization import show_image_coco


def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_val_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


class CocoDatasetAdaptor:
    def __init__(self, images_dir, annotation_filename):
        self.coco_dataset = CocoDetection(
            root=images_dir,
            annFile=annotation_filename
        )

    def __len__(self):
        return len(self.coco_dataset)

    def get_image_and_labels_by_idx(self, index):
        image, ann = self.coco_dataset[index]
        image = np.array(image) / 255.0
        bboxes = np.array([a['bbox'] for a in ann])
        labels = np.array([a['category_id'] for a in ann])
        return image, bboxes, labels, index

    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image_coco(image, list(bboxes))
        print(class_labels)


class EfficientDetDataset(Dataset):
    def __init__(
            self,
            dataset_adaptor,
            transforms
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(pascal_bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)


class EfficientDetDataModule(LightningDataModule):

    def __init__(
            self,
            train_dataset_adaptor,
            validation_dataset_adaptor,
            target_img_size=512,
            num_workers=4,
            batch_size=8
    ):
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.target_img_size = target_img_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds,
            transforms=self.train_transforms()
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds,
            transforms=self.val_transforms()
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader

    def train_transforms(self):
        return get_train_transforms(target_img_size=self.target_img_size)

    def val_transforms(self):
        return get_val_transforms(target_img_size=self.target_img_size)

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids
