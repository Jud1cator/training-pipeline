from argparse import ArgumentParser
from pathlib import Path
from typing import List

import yaml
from albumentations import BasicTransform, CenterCrop, PadIfNeeded, RandomCrop, Resize

from src.utils.typings import ImageResolution

PARSEABLE_TRANSFORMS = [CenterCrop, PadIfNeeded, RandomCrop, Resize]


def create_config_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='Name of the .yml config file in training_pipeline/configs'
    )
    params = parser.parse_args()

    config_path = Path('./configs') / params.config
    with open(config_path, 'r') as fs:
        config = yaml.safe_load(fs)
    return config_path, config


def parse_image_resolution_from_transforms(transform_list: List[BasicTransform]):
    img_res = None
    for tf in transform_list:
        if isinstance(tf, Resize) or isinstance(tf, CenterCrop) or isinstance(tf, RandomCrop):
            img_res = ImageResolution(height=tf.height, width=tf.width)
        elif isinstance(tf, PadIfNeeded):
            img_res = ImageResolution(height=tf.min_height, width=tf.min_width)
    if img_res is None:
        raise RuntimeError(
            f'Transforms must include one of the following to deduce image resolution:'
            f' {[str(t) for t in PARSEABLE_TRANSFORMS]}'
        )
    return img_res
