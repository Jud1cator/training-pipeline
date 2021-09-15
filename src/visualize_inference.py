from argparse import ArgumentParser
from typing import List

import random
import json
import torch
import numpy as np
import cv2
import PIL
from torchvision import transforms as transforms

from models import EfficientDet, get_efficientdet_config


def draw_bboxes(img, bboxes):
    pass


def load_model_for_inference(params):
    model_checkpoint = torch.load(params.model_path)
    network_config = get_efficientdet_config('efficientdet_d0')
    model = EfficientDet(network_config)
    model.load_state_dict(model_checkpoint['state_dict'])
    model.eval()
    return model


def load_annotations(params):
    with open(params.annotations_path, 'r') as fs:
        data = json.load(fs)
    return data


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def prepare_image_for_inference(params, img):
    input_shape = tuple(params.input_shape)
    transform = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor()
    ])
    img = PIL.Image.fromarray(img)
    tensor = transform(img)
    return tensor.unsqueeze(0)


def get_image(params, annotations, idx):
    img_path = params.data_path + '/' + annotations['images'][idx]['file_name']
    img = read_image(img_path)
    return img


def get_gt_bboxes(annotations, idx):
    bboxes = [
        {
            'class': i['category_id'],
            'bbox': i['bbox']
        }
        for i in annotations['annotations']
        if i['image_id'] == idx
    ]
    return bboxes


def get_model_predictions(params, model, img):
    input = prepare_image_for_inference(params, img)
    class_out, box_out = model(input)
    return class_out, box_out


def main(params):
    model = load_model_for_inference(params)
    annotations = load_annotations(params)
    img = get_image(params, annotations, 0)
    gt_bboxes = get_gt_bboxes(annotations, 0)
    class_out, box_out = get_model_predictions(params, model, img)
    print(class_out[0].size(), box_out[0].size())
    print(class_out[1].size(), box_out[1].size())
    print(class_out[2].size(), box_out[2].size())
    print(class_out[3].size(), box_out[3].size())
    print(class_out[4].size(), box_out[4].size())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model_path",
        help="Path to .pth or .json model weights file",
        required=True
    )
    parser.add_argument(
        "-a", "--annotations_path",
        help="Path to .json file with annotations",
        required=True
    )
    parser.add_argument(
        "-d", "--data_path",
        help="Path to directory with images",
        required=True
    )
    parser.add_argument(
        "-i", "--input_shape",
        help="Input shape of the model",
        nargs=2,
        type=int,
        required=True
    )
    parser.add_argument(
        "-o", "--output_path",
        help="Path to output image",
        default="output.png"
    )
    args = parser.parse_args()
    main(args)
