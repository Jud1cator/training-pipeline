import pytorch_lightning as pl
import torch

from src.utils.registry import Registry
from src.losses import DetectionLoss
from src.models.efficientdet.anchors import Anchors, AnchorLabeler


@Registry.register_task
class DetectionTask(pl.LightningModule):

    def __init__(
        self,
        net,
        create_labeler=True,
        optimizer_lr=1e-3,
    ):
        super().__init__()
        self.net = net
        self.lr = optimizer_lr
        self.num_levels = net.config.num_levels
        self.num_classes = net.config.num_classes
        self.anchors = Anchors.from_config(net.config)
        self.max_detection_points = net.config.max_detection_points
        self.max_det_per_image = net.config.max_det_per_image
        self.soft_nms = net.config.soft_nms
        self.anchor_labeler = None
        if create_labeler:
            self.anchor_labeler = AnchorLabeler(
                self.anchors, self.num_classes,
                match_threshold=0.5
            )
        self.loss_fn = DetectionLoss(net.config)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        class_out, box_out = self.model(x)
        if self.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert 'label_num_positives' in target
            cls_targets = [target[f'label_cls_{l}'] for l in
                           range(self.num_levels)]
            box_targets = [target[f'label_bbox_{l}'] for l in
                           range(self.num_levels)]
            num_positives = target['label_num_positives']
        else:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                target['bbox'], target['cls'])

        loss, class_loss, box_loss = self.loss_fn(class_out, box_out,
                                                  cls_targets, box_targets,
                                                  num_positives)
        output = {'loss': loss, 'class_loss': class_loss, 'box_loss': box_loss}

        input, target = batch
        input = torch.stack(input)
        output = self(input, target)
        loss_val = output['loss']
        self.log('train_loss', loss_val)
        return {'loss': loss_val}

    def validation_step(self, batch, batch_idx):
        print(type(batch), len(batch))
        print(type(batch[0]), len(batch[0]))
        input, target = batch
        input = torch.stack(input)
        output = self(input, target)
        loss_val = output['loss']
        self.log('val_loss', loss_val)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', loss_val)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]
