from typing import Dict, List, Optional, Tuple

from effdet.bench import _post_process
from torch import Tensor

from src.losses.detection_loss import create_detection_loss
from src.models.detection.utils.anchors import AnchorLabeler, Anchors
from src.models.detection.utils.helpers import jit_batch_detection
from src.models.utils.abstract_model_wrapper import AbstractModelWrapper


class DetectionModelWrapper(AbstractModelWrapper):

    def __init__(
            self,
            num_classes: int,
            anchor_params: dict,
            max_detection_points: int,
            soft_nms: bool,
            max_det_per_image: int
    ):
        super().__init__()
        # Add class for background
        self.num_classes = num_classes + 1
        self.anchors = Anchors(**anchor_params)
        self.num_levels = self.anchors.max_level - self.anchors.min_level + 1
        self.anchor_labeler = AnchorLabeler(self.anchors, self.num_classes)
        self.loss_fn = create_detection_loss(self.num_classes)

        self.max_detection_points = max_detection_points
        self.soft_nms = soft_nms
        self.max_det_per_image = max_det_per_image

    def infer(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        pass

    def get_detections_from_batch(
            self,
            batch_size: int,
            class_out: List[Tensor],
            box_out: List[Tensor]
    ):
        class_out, box_out, indices, classes = _post_process(
            class_out, box_out,
            num_levels=self.num_levels, num_classes=self.num_classes,
            max_detection_points=self.max_detection_points
        )

        return jit_batch_detection(
            batch_size, class_out, box_out, self.anchors.boxes, indices, classes,
            max_det_per_image=self.max_det_per_image, soft_nms=self.soft_nms
        )

    def predict(self, x: List[Tensor]):
        class_out, box_out = self.infer(x)
        batch_size = len(x)
        return self.get_detections_from_batch(batch_size, class_out, box_out)

    def forward(self, x: List[Tensor], target: Optional[List[Dict[str, Tensor]]] = None):
        if target is None:
            return self.predict(x)
        else:
            true_boxes = [t['bboxes'].float() for t in target]
            true_labels = [t['labels'].float() for t in target]
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                true_boxes, true_labels
            )
            class_out, box_out = self.infer(x)
            loss, class_loss, box_loss = self.loss_fn(
                class_out, box_out, cls_targets, box_targets, num_positives)
            output = {'loss': loss, 'class_loss': class_loss, 'box_loss': box_loss}
            if not self.training:
                # if eval mode, output detections for evaluation
                batch_size = len(x)
                output['detections'] = self.get_detections_from_batch(
                    batch_size, class_out, box_out
                )
        return output
