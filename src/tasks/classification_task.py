import pytorch_lightning as pl
import torch
import numpy as np

from registry import Registry
from metrics import ConfusionMatrix
from data_modules import ClassificationDataModule
from utils.visualization import visualize_batch, plot_confusion_matrix


class ClassificationTask(pl.LightningModule):

    def __init__(
            self,
            datamodule: ClassificationDataModule,
            network: dict,
            loss: dict,
            optimizer: dict,
            scheduler: dict,
            debug: bool = False,
            res_dir=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.classes = datamodule.classes
        self.num_classes = len(self.classes)

        class_weights = datamodule.class_weights if loss['weighted'] else None
        print(class_weights)
        self.loss = Registry.LOSSES[loss['name']](
            weight=class_weights, **loss['params']
        )

        self.optimizer_dict = optimizer
        self.scheduler_dict = scheduler

        self.net = Registry.MODELS[network['name']](
            num_classes=self.num_classes,
            input_shape=datamodule.input_shape,
            **network['params']
        ).get_model()

        self.cm = ConfusionMatrix(self.num_classes)

        self.visualize_train = debug
        self.visualize_val = debug
        self.visualize_test = debug

        self.res_dir = res_dir

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, true = batch
        if self.visualize_train:
            visualize_batch(img)
            self.visualize_train = False
        pred = self(img)
        loss_val = self.loss(pred, true)
        self.log('train_loss', loss_val)
        return {'loss': loss_val}

    def validation_step(self, batch, batch_idx):
        img, true = batch
        if self.visualize_val:
            visualize_batch(img)
            self.visualize_val = False
        pred = self(img)
        loss_val = self.loss(pred, true)
        self.log('val_loss', loss_val)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', loss_val)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        img, true = batch
        if self.visualize_test:
            visualize_batch(img)
            self.visualize_test = False
        pred = np.argmax(self(img).cpu().numpy(), axis=1)
        self.cm.update(true.cpu().numpy(), pred)

    def test_epoch_end(self, outputs) -> None:
        cm = self.cm.get_confusion_matrix()
        precision = self.cm.get_precision()
        recall = self.cm.get_recall()
        avg_precision = precision.sum() / len(precision)
        avg_recall = recall.sum() / len(recall)
        self.log('average_precision', avg_precision)
        self.log('average_recall', avg_recall)
        plot_confusion_matrix(
            cm,
            path=self.res_dir/'confusion_matrix.png',
            categories=self.classes,
            sort=False
        )

    def configure_optimizers(self):
        return_dict = dict()
        return_dict['optimizer'] = Registry.OPTIMIZERS[self.optimizer_dict['name']](
            self.net.parameters(), **self.optimizer_dict['params']
        )
        if self.scheduler_dict:
            return_dict['scheduler'] = Registry.SCHEDULERS[self.scheduler_dict['name']](
                return_dict['optimizer'], **self.scheduler_dict['params']
            )
            return_dict['monitor'] = 'val_loss'
        return return_dict
