import torch
import matplotlib.pyplot as plt

from pytorch_lightning import Trainer
from torchvision.datasets import CocoDetection
from torch.optim import Adam

from data_modules.efficientdet_datamodule import CocoDatasetAdaptor, EfficientDetDataModule
from data_modules.detection_datamodule import DetectionDataModule
from models.mobiledets import MobileDetCPU
from tasks.efficientdet_task import EfficientDetModel
from tasks.detection_task import DetectionTask
from utils.visualization import draw_coco_bboxes
from registry import Registry


if __name__ == '__main__':

    Registry.init_modules()

    img_size = 384

    datamodule = DetectionDataModule(
        images_dir='path to images',
        annotation_files='path to annotation file',
        batch_size=2,
        image_size=[img_size, img_size],
        val_split=0.1,
        test_split=0.8
    )

    task = DetectionTask(
        datamodule=datamodule,
        model={
            'name': 'RossMobileDetV2',
            'params': {
                'num_classes': 1,
                'pretrained': True
            }
        },
        optimizer={'name': 'Adam', 'params': {'lr': 0.001}},
        debug=True
    )

    trainer = Trainer(gpus=0, max_epochs=2)
    trainer.fit(task, datamodule=datamodule)