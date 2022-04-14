import logging
import re
from typing import Any, Dict, Iterable, List, Optional
from warnings import warn

import torch
from pl_bolts.models.detection.yolo.yolo_config import _create_layer
from pl_bolts.models.detection.yolo.yolo_layers import DetectionLayer, RouteLayer, ShortcutLayer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn
from torchvision.ops import nms

from src.models.utils.abstract_model_wrapper import AbstractModelWrapper

log = logging.getLogger(__name__)


class YOLOv4TinyWrapper(AbstractModelWrapper):
    def __init__(
            self,
            num_classes,
            confidence_threshold,
            nms_threshold,
            max_predictions_per_image
    ):
        super().__init__()
        # self.module_list = YOLOConfiguration('yolo_configs/yolov4-tiny.cfg').get_network()
        sections = self._read_file(yolov4_tiny_cfg.split(sep='\n'))

        if len(sections) < 2:
            raise MisconfigurationException(
                'The model configuration file should include at least two sections.'
            )

        self.__dict__.update(sections[0])
        self.global_config = sections[0]
        self.layer_configs = sections[1:]

        self.module_list = nn.ModuleList()
        num_inputs = [3]  # Number of channels in the input of every layer up to the current layer
        for layer_config in self.layer_configs:
            config = {**self.global_config, **layer_config}
            if config['type'] == 'yolo':
                config['classes'] = num_classes + 1
            module, num_outputs = _create_layer(config, num_inputs)
            self.module_list.append(module)
            num_inputs.append(num_outputs)

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_predictions_per_image = max_predictions_per_image

    @staticmethod
    def _read_file(config_file: Iterable[str]) -> List[Dict[str, Any]]:
        """Reads a YOLOv4 network configuration file and returns a list of configuration sections.

        Args:
            config_file: The configuration file to read.

        Returns:
            A list of configuration sections.
        """
        section_re = re.compile(r'\[([^]]+)\]')
        list_variables = ('layers', 'anchors', 'mask', 'scales')
        variable_types = {
            'activation': str,
            'anchors': int,
            'angle': float,
            'batch': int,
            'batch_normalize': bool,
            'beta_nms': float,
            'burn_in': int,
            'channels': int,
            'classes': int,
            'cls_normalizer': float,
            'decay': float,
            'exposure': float,
            'filters': int,
            'from': int,
            'groups': int,
            'group_id': int,
            'height': int,
            'hue': float,
            'ignore_thresh': float,
            'iou_loss': str,
            'iou_normalizer': float,
            'iou_thresh': float,
            'jitter': float,
            'layers': int,
            'learning_rate': float,
            'mask': int,
            'max_batches': int,
            'max_delta': float,
            'momentum': float,
            'mosaic': bool,
            'new_coords': int,
            'nms_kind': str,
            'num': int,
            'obj_normalizer': float,
            'pad': bool,
            'policy': str,
            'random': bool,
            'resize': float,
            'saturation': float,
            'scales': float,
            'scale_x_y': float,
            'size': int,
            'steps': str,
            'stride': int,
            'subdivisions': int,
            'truth_thresh': float,
            'width': int,
        }

        section = None
        sections = []

        def convert(key, value):
            """Converts a value to the correct type based on key."""
            if key not in variable_types:
                warn('Unknown YOLO configuration variable: ' + key)
                return key, value
            if key in list_variables:
                value = [variable_types[key](v) for v in value.split(',')]
            else:
                value = variable_types[key](value)
            return key, value

        for line in config_file:
            line = line.strip()
            if (not line) or (line[0] == '#'):
                continue

            section_match = section_re.match(line)
            if section_match:
                if section is not None:
                    sections.append(section)
                section = {'type': section_match.group(1)}
            else:
                key, value = line.split('=')
                key = key.rstrip()
                value = value.lstrip()
                key, value = convert(key, value)
                section[key] = value
        if section is not None:
            sections.append(section)

        return sections

    def forward(self, x: List[torch.Tensor], target: Optional[List[Dict[str, Tensor]]] = None):
        """
        Runs a forward pass through the network (all layers listed in ``self.network``), and if
        training targets are provided, computes the losses from the detection layers.

        Detections are concatenated from the detection layers. Each image will produce
        `N * num_anchors * grid_height * grid_width` detections, where `N` depends on the number of
        detection layers. For one detection layer `N = 1`, and each detection layer increases it by
        a number that depends on the size of the feature map on that layer. For example, if the
        feature map is twice as wide and high as the grid, the layer will add four times more
        features.

        Args:
            images: Images to be processed. Tensor of size
                ``[batch_size, num_channels, height, width]``.
            target: If set, computes losses from detection layers against these targets. A list of
                dictionaries, one for each image.

        Returns:
            detections (:class:`~torch.Tensor`), losses (Dict[str, :class:`~torch.Tensor`]):
            Detections, and if targets were provided, a dictionary of losses. Detections are shaped
            ``[batch_size, num_predictors, num_classes + 5]``, where ``num_predictors`` is the
            total number of cells in all detection layers times the number of boxes predicted by
            one cell. The predicted box coordinates are in `(x1, y1, x2, y2)` format and scaled to
            the input image size.
        """
        x = torch.stack(x)

        outputs = []  # Outputs from all layers
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers
        hits = []  # Number of targets each detection layer was responsible for

        image_height = x.shape[2]
        image_width = x.shape[3]
        image_size = torch.tensor([image_width, image_height], device=x.device)

        for module in self.module_list:
            if isinstance(module, (RouteLayer, ShortcutLayer)):
                x = module(x, outputs)
            elif isinstance(module, DetectionLayer):
                if target is None:
                    x = module(x, image_size)
                    if not self.training:
                        detections.append(x)
                else:
                    for t in target:
                        t['boxes'] = t['bboxes']
                        t['labels'] = t['labels'].long()
                    x, layer_losses, layer_hits = module(x, image_size, target)
                    if not self.training:
                        detections.append(x)
                    losses.append(layer_losses)
                    hits.append(layer_hits)
            else:
                x = module(x)

            outputs.append(x)

        if not self.training:
            detections = torch.cat(detections, 1)
            detections = self._split_detections(detections)
            detections = self._filter_detections(detections)
            values = []
            for i in range(len(detections['boxes'])):
                values.append(
                    torch.cat(
                        [
                            detections['boxes'][i],
                            detections['scores'][i].unsqueeze(1),
                            detections['labels'][i].unsqueeze(1)
                        ],
                        dim=1
                    )
                )
            detections = torch.stack(values)
            if target is None:
                return detections

        total_hits = sum(hits)
        num_targets = sum(len(image_targets['boxes']) for image_targets in target)
        if total_hits != num_targets:
            warn(
                f'{num_targets} training targets were matched a total of {total_hits} times by'
                f' detection layers. Anchors may have been configured incorrectly.'
            )
        for layer_idx, layer_hits in enumerate(hits):
            hit_rate = torch.true_divide(layer_hits, total_hits) if total_hits > 0 else 1.0
            self.log(f'layer_{layer_idx}_hit_rate', hit_rate, sync_dist=False)

        def total_loss(loss_name):
            """Returns the sum of the loss over detection layers."""
            loss_tuple = tuple(layer_losses[loss_name] for layer_losses in losses)
            return torch.stack(loss_tuple).sum()

        output = {loss_name: total_loss(loss_name) for loss_name in losses[0].keys()}
        output['box_loss'] = 40.0 * output['overlap']
        output['class_loss'] = output['class']
        output['loss'] = torch.stack([output['box_loss'], output['class_loss']]).sum()
        output.pop('overlap')
        output.pop('class')
        if not self.training:
            output['detections'] = detections
        return output

    @staticmethod
    def _split_detections(detections: Tensor) -> Dict[str, Tensor]:
        """Splits the detection tensor returned by a forward pass into a dictionary.

        The fields of the dictionary are as follows:
            - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)`
                coordinates
            - scores (``Tensor[batch_size, N]``): detection confidences
            - classprobs (``Tensor[batch_size, N]``): probabilities of the best classes
            - labels (``Int64Tensor[batch_size, N]``): the predicted labels for each image

        Args:
            detections: A tensor of detected bounding boxes and their attributes.

        Returns:
            A dictionary of detection results.
        """
        boxes = detections[..., :4]
        scores = detections[..., 4]
        classprobs = detections[..., 5:]
        classprobs, labels = torch.max(classprobs, -1)
        return {'boxes': boxes, 'scores': scores, 'classprobs': classprobs, 'labels': labels}

    def _filter_detections(self, detections: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        """
        Filters detections based on confidence threshold. Then for every class performs non-maximum
        suppression (NMS). NMS iterates the bounding boxes that predict this class in descending
        order of confidence score, and removes lower scoring boxes that have an IoU greater than the
        NMS threshold with a higher scoring box. Finally the detections are sorted by descending
        confidence and possible truncated to the maximum number of predictions.

        Args:
            detections: All detections. A dictionary of tensors, each containing the predictions
                from all images.

        Returns:
            Filtered detections. A dictionary of lists, each containing a tensor per image.
        """
        boxes = detections['boxes']
        scores = detections['scores']
        classprobs = detections['classprobs']
        labels = detections['labels']

        out_boxes = []
        out_scores = []
        out_classprobs = []
        out_labels = []

        for img_boxes, img_scores, img_classprobs, img_labels in zip(
                boxes, scores, classprobs, labels
        ):
            # Select detections with high confidence score.
            selected = img_scores > self.confidence_threshold
            img_boxes = img_boxes[selected]
            img_scores = img_scores[selected]
            img_classprobs = img_classprobs[selected]
            img_labels = img_labels[selected]

            img_out_boxes = boxes.new_zeros((0, 4))
            img_out_scores = scores.new_zeros(0)
            img_out_classprobs = classprobs.new_zeros(0)
            img_out_labels = labels.new_zeros(0)

            # Iterate through the unique object classes detected in the image and perform NMS
            # for the objects of the class in question.
            for cls_label in labels.unique():
                selected = img_labels == cls_label
                cls_boxes = img_boxes[selected]
                cls_scores = img_scores[selected]
                cls_classprobs = img_classprobs[selected]
                cls_labels = img_labels[selected]

                # NMS will crash if there are too many boxes.
                cls_boxes = cls_boxes[:100000]
                cls_scores = cls_scores[:100000]
                selected = nms(cls_boxes, cls_scores, self.nms_threshold)

                img_out_boxes = torch.cat((img_out_boxes, cls_boxes[selected]))
                img_out_scores = torch.cat((img_out_scores, cls_scores[selected]))
                img_out_classprobs = torch.cat((img_out_classprobs, cls_classprobs[selected]))
                img_out_labels = torch.cat((img_out_labels, cls_labels[selected]))

            # Sort by descending confidence and limit the maximum number of predictions.
            indices = torch.argsort(img_out_scores, descending=True)
            if self.max_predictions_per_image >= 0:
                indices = indices[: self.max_predictions_per_image]
            out_boxes.append(img_out_boxes[indices])
            out_scores.append(img_out_scores[indices])
            out_classprobs.append(img_out_classprobs[indices])
            out_labels.append(img_out_labels[indices])

        return {
            'boxes': out_boxes,
            'scores': out_scores,
            'classprobs': out_classprobs,
            'labels': out_labels
        }


yolov4_tiny_cfg = \
    """
[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000

max_batches = 2000200
policy=steps
steps=1600000,1800000
scales=.1,.1


#weights_reject_freq=1001
#ema_alpha=0.9998
#equidistant_point=1000
#num_sigmas_reject_badlabels=3
#badlabels_rejection_percentage=0.2


[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear



[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6
#new_coords=1
#scale_x_y = 2.0

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 23

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear

[yolo]
mask = 1,2,3
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6
#new_coords=1
#scale_x_y = 2.0
"""
