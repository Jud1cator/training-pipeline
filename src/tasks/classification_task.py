import pytorch_lightning as pl
import torch
import numpy as np

from registry import Registry
from metrics import ConfusionMatrix
from data_modules import ClassificationDataModule
from utils.config_validation import Config
from utils.visualization import visualize_batch
from utils.visualization import plot_confusion_matrix


class ClassificationTask(pl.LightningModule):

    def __init__(
            self,
            datamodule: ClassificationDataModule,
            network: dict,
            loss: dict,
            metrics: list,
            optimizer: dict,
            scheduler: dict = None,
            debug: bool = False,
            res_dir=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.classes = datamodule.classes
        self.num_classes = len(self.classes)

        network = Config(**network)
        loss = Config(**loss)
        self.optimizer_config = Config(**optimizer)
        self.scheduler_dict = Config(**scheduler) if scheduler else None

        self.net = Registry.MODELS[network.name](
            num_classes=self.num_classes, **network.params).get_model()

        class_weights = datamodule.class_weights if loss.params['is_weighted'] else None
        loss.params.pop('is_weighted')
        self.loss = Registry.LOSSES[loss.name](weight=class_weights, **loss.params)

        self.confusion_matrix = ConfusionMatrix(self.num_classes)
        self.metrics = {}
        for metric in metrics:
            metric = Config(**metric)
            self.metrics[metric.name] = Registry.METRICS[metric.name](**metric.params)

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
        output = self(img)
        loss = self.loss(output, true)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, logger=False, prog_bar=True)
        self.logger.log_metrics({'train_loss': loss}, step=self.current_epoch)

    def validation_epoch_start(self):
        self.confusion_matrix.reset()

    def validation_step(self, batch, batch_idx):
        img, target = batch
        if self.visualize_val:
            visualize_batch(img)
            self.visualize_val = False
        output = self(img)
        loss = self.loss(output, target)
        prediction = np.argmax(output.cpu().numpy(), axis=1)
        self.confusion_matrix.update(prediction, target.cpu().numpy())
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', loss, logger=False, prog_bar=True)
        self.logger.log_metrics({'val_loss': loss}, step=self.current_epoch)
        for metric_name, metric in self.metrics.items():
            key = '_'.join(['val', metric_name.lower()])
            value = metric.get_value(self.confusion_matrix.get_value())
            self.log(key, value, logger=False, prog_bar=True)
            self.logger.log_metrics({key: value}, step=self.current_epoch)

    def test_epoch_start(self):
        self.confusion_matrix.reset()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        img, target = batch
        if self.visualize_test:
            visualize_batch(img)
            self.visualize_test = False
        prediction = np.argmax(self(img).cpu().numpy(), axis=1)
        self.confusion_matrix.update(prediction, target.cpu().numpy())

    def test_epoch_end(self, outputs) -> None:
        save_path = self.res_dir / 'confusion_matrix.png' if self.res_dir else None
        plot_confusion_matrix(
            self.confusion_matrix.get_value(),
            categories=self.classes,
            save_path=save_path,
            sort=False,
            show=True
        )
        for metric_name, metric in self.metrics.items():
            key = '_'.join(['test', metric_name.lower()])
            value = metric.get_value(self.confusion_matrix.get_value())
            self.logger.log_metrics({key: value}, step=self.current_epoch)

    def configure_optimizers(self):
        config = {}
        opt = Registry.OPTIMIZERS[self.optimizer_config.name](
            self.net.parameters(), **self.optimizer_config.params
        )
        config['optimizer'] = opt
        if self.scheduler_dict:
            sch = Registry.SCHEDULERS[self.scheduler_config.name](
                opt, **self.scheduler_config.params)
            config['lr_scheduler'] = {
                'scheduler': sch,
                'monitor': 'val_loss'
            }
        return config
