from typing import List

from ensemble_boxes import ensemble_boxes_wbf
import numpy as np
import pytorch_lightning as pl
import torch

from metrics.coco_metrics import get_coco_stats
from registry import Registry
from utils.config_validation import Config
from utils.visualization import visualize_batch, visualize_with_boxes


def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels


@Registry.register_task
class DetectionTask(pl.LightningModule):

    def __init__(
            self,
            datamodule,
            model: dict,
            optimizer: dict,
            scheduler: dict = None,
            debug: bool = False,
            **kwargs
    ):
        super().__init__()

        self.img_size = datamodule.image_size[0]

        model_config = Config(**model)
        self.optimizer_config = Config(**optimizer)
        self.scheduler_dict = Config(**scheduler) if scheduler else None

        self.model = Registry.MODELS[model_config.name](**model_config.params).get_model()

        self.visualize_train = debug
        self.visualize_val = debug
        self.visualize_test = debug

    def forward(self, x, target):
        # target = {'bbox': true_boxes, 'cls': true_labels, 'img_scale': None, 'img_size': None}
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

    def training_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        if self.visualize_train:
            visualize_batch(images)
            # visualize_with_boxes(images, true_boxes, true_labels)
            self.visualize_train = False
        output = self.forward(images, annotations)
        return output

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        box_loss = torch.stack([x['box_loss'] for x in outputs]).mean()
        class_loss = torch.stack([x['class_loss'] for x in outputs]).mean()
        metrics = {'loss': loss, 'box_loss': box_loss, 'class_loss': class_loss}
        self.log_metrics(metrics, prefix='train')

    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        if self.visualize_val:
            visualize_batch(images)
            # visualize_with_boxes(images, true_boxes, true_labels)
            self.visualize_val = False
        output = self.forward(images, annotations)
        batch_predictions = {
            'predictions': output['detections'],
            'targets': targets,
            'image_ids': image_ids
        }
        self.log_loss(output, prefix='val')
        output_dict = {f'val_{k}': v for k, v in output.items()}
        output_dict['batch_predictions'] = batch_predictions
        return output_dict

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        box_loss = torch.stack([x['val_box_loss'] for x in outputs]).mean()
        class_loss = torch.stack([x['val_class_loss'] for x in outputs]).mean()

        pred_labels, image_ids, pred_bboxes, pred_confidences, targets = \
            self.aggregate_prediction_outputs(outputs)
        truth_image_ids = [target["image_id"].detach().item() for target in targets]
        truth_boxes = [
            target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
        ]  # convert to xyxy for evaluation
        truth_labels = [target["labels"].detach().tolist() for target in targets]

        stats = get_coco_stats(
            prediction_image_ids=image_ids,
            predicted_class_confidences=pred_confidences,
            predicted_bboxes=pred_bboxes,
            predicted_class_labels=pred_labels,
            target_image_ids=truth_image_ids,
            target_bboxes=truth_boxes,
            target_class_labels=truth_labels
        )['All']

        metrics = {'loss': loss, 'box_loss': box_loss, 'class_loss': class_loss, 'AP': stats['AP_all']}
        self.log_metrics(metrics, prefix='val')

    def test_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        if self.visualize_test:
            visualize_batch(images)
            # visualize_with_boxes(images, true_boxes, true_labels)
            self.visualize_test = False
        output = self.forward(images, annotations)
        batch_predictions = {
            'predictions': output['detections'],
            'targets': targets,
            'image_ids': image_ids
        }
        output_dict = {f'val_{k}': v for k, v in output.items()}
        output_dict['batch_predictions'] = batch_predictions
        return output_dict

    def test_epoch_end(self, outputs):
        pred_labels, image_ids, pred_bboxes, pred_confidences, targets = \
            self.aggregate_prediction_outputs(outputs)
        target_image_ids = [target["image_id"].detach().item() for target in targets]
        target_boxes = [
            target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
        ]  # convert to xyxy for evaluation
        target_labels = [target["labels"].detach().tolist() for target in targets]

        stats = get_coco_stats(
            prediction_image_ids=image_ids,
            predicted_class_confidences=pred_confidences,
            predicted_bboxes=pred_bboxes,
            predicted_class_labels=pred_labels,
            target_image_ids=target_image_ids,
            target_bboxes=target_boxes,
            target_class_labels=target_labels
        )['All']

        metrics = {'AP': stats['AP_all']}
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

    def predict(self, images):
        if isinstance(images, list):
            return self.predict_list(images)
        elif isinstance(images, torch.Tensor):
            return self.predict_tensor(images)
        else:
            raise TypeError(f'Unsupported images type: {type(images)}')

    def predict_list(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    def predict_tensor(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
                images_tensor.shape[-1] != self.img_size
                or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self._rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences

    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=0.5, skip_box_thr=0.1
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > 0.1)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def _rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                            np.array(bboxes)
                            * [
                                im_w / self.img_size,
                                im_h / self.img_size,
                                im_w / self.img_size,
                                im_h / self.img_size,
                            ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

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
