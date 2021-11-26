from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List


__all__ = ["get_stats_at_annotation_level", "get_coco_stats"]


def convert_pascal_bbox_to_coco(xmin, ymin, xmax, ymax):
    """
    pascal: top-left-x, top-left-y, x-bottom-right, y-bottom-right
    coco:   top-left-x, top-left-y, width and height
    """
    return [xmin, ymin, xmax - xmin, ymax - ymin]


class AMLCOCO(COCO):
    def __init__(self, annotation_gt=None):

        if annotation_gt is None or type(annotation_gt) == str:
            COCO.__init__(self, annotation_file=annotation_gt)
        else:
            COCO.__init__(self, annotation_file=None)

            self.dataset = annotation_gt
            self.createIndex()


def get_stats_dict(stats=None, summ_type="bbox"):
    if summ_type == "bbox":
        if stats is None:
            stats = [-1] * 12
        r = {
            "AP_all": stats[0],
            "AP_all_IOU_0_50": stats[1],
            "AP_all_IOU_0_75": stats[2],
            "AP_small": stats[3],
            "AP_medium": stats[4],
            "AP_large": stats[5],
            "AR_all_dets_1": stats[6],
            "AR_all_dets_10": stats[7],
            "AR_all": stats[8],
            "AR_small": stats[9],
            "AR_medium": stats[10],
            "AR_large": stats[11],
        }
        return r
    return None


def convert_image_ids_to_coco(image_ids):
    img_ids = set(image_ids)
    images = []
    for img_id in img_ids:
        images.extend(
            [
                {
                    "id": img_id,
                },
            ]
        )
    return images


def convert_class_labels_to_coco_cats(class_labels):

    cat_set = set()
    for label_list in class_labels:
        for label in label_list:
            cat_set.add(label)

    cats = []
    for cat in cat_set:
        cats.extend(
            [
                {
                    "id": cat,
                }
            ]
        )

    return cats


def convert_ground_truth_to_coco_annots(
    target_image_ids, target_class_labels, target_bboxes, convert_bbox_func=None
):

    # conv bbox to coco annotation
    annots = []
    ann_id = 1
    for target_image_id, ground_truth_boxes, ground_truth_labels in zip(
        target_image_ids, target_bboxes, target_class_labels
    ):
        for bbox, label in zip(ground_truth_boxes, ground_truth_labels):

            if convert_bbox_func:
                coco_bbox = convert_bbox_func(*bbox)
            else:
                coco_bbox = bbox

            annots.extend(
                [
                    {
                        "id": ann_id,
                        "bbox": coco_bbox,  # coco format: x, y, w, h
                        "category_id": label,
                        "image_id": target_image_id,
                        "iscrowd": 0,
                        "area": coco_bbox[2] * coco_bbox[3],
                    }
                ]
            )

            ann_id += 1

    return annots


def create_ground_truth(
    target_image_ids, target_class_labels, target_bboxes, convert_bbox_func=None
):

    cats = convert_class_labels_to_coco_cats(target_class_labels)

    images = convert_image_ids_to_coco(target_image_ids)

    annots = convert_ground_truth_to_coco_annots(
        target_image_ids, target_class_labels, target_bboxes, convert_bbox_func
    )

    return {"images": images, "annotations": annots, "categories": cats}


def create_detections(
    prediction_image_ids,
    predicted_class_confidences,
    predicted_class_labels,
    predicted_bboxes,
    convert_bbox_func=None,
):

    detections = []
    for image_id, class_predictions, confidences, box_predictions in zip(
        prediction_image_ids,
        predicted_class_labels,
        predicted_class_confidences,
        predicted_bboxes,
    ):
        # add prediction boxes
        for class_prediction, class_prediction_confidence, bbox in zip(
            class_predictions, confidences, box_predictions
        ):
            if convert_bbox_func:
                coco_bbox = convert_bbox_func(*bbox)
            else:
                coco_bbox = bbox

            detections.extend(
                [
                    {
                        "bbox": coco_bbox,  # coco format: x, y, w, h
                        "category_id": class_prediction,
                        "score": class_prediction_confidence,
                        "image_id": image_id,
                    }
                ]
            )

    return detections


def get_stats_at_annotation_level(
    predicted_class_labels: List[List[int]],
    predicted_class_confidences: List[List[float]],
    predicted_bboxes: List[List[List[float]]],
    prediction_image_ids: List[int],
    target_image_ids: List[int],
    target_class_labels: List[List[int]],
    target_bboxes: List[List[List[float]]],
    convert_bbox_func=convert_pascal_bbox_to_coco,
):
    """
    :param predicted_class_labels: A list containing a list of class lolabels predicted per image
    :param predicted_class_confidences: A list containing a list of prediction confidence values per image
    :param predicted_bboxes: A list containing a list of bounding boxes, in Pascal VOC format, predicted per image
    :param prediction_image_ids: A list of image ids for each image in the prediction lists
    :param target_image_ids: A list of image ids for each image in the target lists
    :param target_class_labels: A list containing a list of ground truth class labels per image
    :param target_bboxes: A list containing a list of ground truth bounding boxes, in Pascal VOC format
    :param convert_bbox_func: A function to convert the format of incoming bboxes to coco format, can set to None
    :returns: a dictionary of the coco results. Returns all -1s if there are no predictions.
    """

    results = get_coco_stats(
        predicted_class_labels,
        predicted_class_confidences,
        predicted_bboxes,
        prediction_image_ids,
        target_image_ids,
        target_class_labels,
        target_bboxes,
        convert_bbox_func=convert_bbox_func,
    )

    return results["All"]


def get_coco_stats(
    predicted_class_labels: List[List[int]],
    predicted_class_confidences: List[List[float]],
    predicted_bboxes: List[List[List[float]]],
    prediction_image_ids: List[int],
    target_image_ids: List[int],
    target_class_labels: List[List[int]],
    target_bboxes: List[List[List[float]]],
    convert_bbox_func=convert_pascal_bbox_to_coco,
    include_per_class=False,
):
    """
    :param predicted_class_labels: A list containing a list of class labels predicted per image
    :param predicted_class_confidences: A list containing a list of prediction confidence values per image
    :param predicted_bboxes: A list containing a list of bounding boxes, in Pascal VOC format, predicted per image
    :param prediction_image_ids: A list of image ids for each image in the prediction lists
    :param target_image_ids: A list of image ids for each image in the target lists
    :param target_class_labels: A list containing a list of ground truth class labels per image
    :param target_bboxes: A list containing a list of ground truth bounding boxes, in Pascal VOC format
    :param convert_bbox_func: A function to convert the format of incoming bboxes to coco format, can set to None
    :param include_per_class: Calculate and return per class result
    :returns: a dictionary of the coco results. Returns all -1s if there are no predictions.
    """

    results = {}

    # create coco result dictionary from predictions
    dt = create_detections(
        prediction_image_ids,
        predicted_class_confidences,
        predicted_class_labels,
        predicted_bboxes,
        convert_bbox_func=convert_bbox_func,
    )

    if len(dt) == 0:
        # no predictions so return all -1s.
        results["All"] = get_stats_dict(stats=None)
        return results

    # create coco dict for the ground truth
    gt = create_ground_truth(
        target_image_ids,
        target_class_labels,
        target_bboxes,
        convert_bbox_func=convert_bbox_func,
    )

    # load the coco dictionaries
    coco_gt = AMLCOCO(annotation_gt=gt)
    coco_dt = coco_gt.loadRes(dt)

    # do the eval
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    image_ids = coco_gt.getImgIds()
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results["All"] = get_stats_dict(coco_eval.stats)

    if include_per_class:
        class_labels = coco_gt.getCatIds()
        for class_label in class_labels:
            coco_eval.params.catIds = [class_label]
            image_ids = coco_gt.getImgIds()
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            results[class_label] = get_stats_dict(coco_eval.stats)

    return results
