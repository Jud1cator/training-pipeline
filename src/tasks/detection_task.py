import pytorch_lightning as pl
import torch

from registry import Registry
from utils.config_validation import Config
from utils.visualization import visualize_batch, visualize_with_boxes


@Registry.register_task
class DetectionTask(pl.LightningModule):

    def __init__(
            self,
            model: dict,
            optimizer: dict,
            scheduler: dict = None,
            debug: bool = False,
            **kwargs
    ):
        super().__init__()

        model_config = Config(**model)
        self.optimizer_config = Config(**optimizer)
        self.scheduler_dict = Config(**scheduler) if scheduler else None

        self.model = Registry.MODELS[model_config.name](**model_config.params).get_model()

        self.visualize_train = debug
        self.visualize_val = debug
        self.visualize_test = debug

    def forward(self, x, true_boxes, true_labels):
        target = {'bbox': true_boxes, 'cls': true_labels, 'img_scale': None, 'img_size': None}
        return self.model(x, target)

    def log_loss(self, loss_dict, prefix=''):
        for name, item in loss_dict.items():
            if name == 'detections':
                continue
            name = f'{prefix}_{name}'
            self.log(name, item, logger=False, prog_bar=True)

    def log_metrics(self, metrics_dict, prefix=''):
        for name, item in metrics_dict.items():
            if name == 'detections':
                continue
            name = f'{prefix}_{name}'
            self.logger.log_metrics({name: item}, step=self.current_epoch)

    def aggregate_prediction_outputs(self, outputs):

        detections = torch.cat([output["batch_predictions"]["predictions"] for output in outputs])

        image_ids = []
        targets = []
        for output in outputs:
            batch_predictions = output["batch_predictions"]
            image_ids.extend(batch_predictions["image_ids"])
            targets.extend(batch_predictions["targets"])

        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels
        ) = self.post_process_detections(detections)

        return (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets,
        )

    def training_step(self, batch, batch_idx):
        images, true_boxes, true_labels = batch
        if self.visualize_train:
            visualize_batch(images)
            # visualize_with_boxes(images, true_boxes, true_labels)
            self.visualize_train = False
        output = self.forward(images, true_boxes, true_labels)
        return output

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        box_loss = torch.stack([x['box_loss'] for x in outputs]).mean()
        class_loss = torch.stack([x['class_loss'] for x in outputs]).mean()
        metrics = {'loss': loss, 'box_loss': box_loss, 'class_loss': class_loss}
        self.log_metrics(metrics, prefix='train')

    def validation_step(self, batch, batch_idx):
        images, true_boxes, true_labels = batch
        if self.visualize_val:
            visualize_batch(images)
            # visualize_with_boxes(images, true_boxes, true_labels)
            self.visualize_val = False
        output = self.forward(images, true_boxes, true_labels)
        self.log_loss(output, prefix='val')
        return {f'val_{k}': v for k, v in output.items()}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        box_loss = torch.stack([x['val_box_loss'] for x in outputs]).mean()
        class_loss = torch.stack([x['val_class_loss'] for x in outputs]).mean()
        metrics = {'loss': loss, 'box_loss': box_loss, 'class_loss': class_loss}
        self.log_metrics(metrics, prefix='val')

    def configure_optimizers(self):
        config = {}
        opt = Registry.OPTIMIZERS[self.optimizer_config.name](
            self.model.parameters(), **self.optimizer_config.params
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
