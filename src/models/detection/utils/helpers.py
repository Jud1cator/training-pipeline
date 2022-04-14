from typing import Optional

import torch
from effdet.anchors import generate_detections


@torch.jit.script
def jit_batch_detection(
        batch_size: int, class_out, box_out, anchor_boxes, indices, classes,
        img_scale: Optional[torch.Tensor] = None,
        img_size: Optional[torch.Tensor] = None,
        max_det_per_image: int = 100,
        soft_nms: bool = False,
):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):

        scale = img_scale
        size = img_size

        img_scale_i = None if scale is None else scale[i]
        img_size_i = None if size is None else size[i]

        detections = generate_detections(
            class_out[i], box_out[i], anchor_boxes, indices[i], classes[i],
            img_scale_i, img_size_i, max_det_per_image=max_det_per_image, soft_nms=soft_nms)
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)
