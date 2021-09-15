import yaml
from argparse import ArgumentParser
from pathlib import Path


def create_config_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Name of the .yml config file in training_pipeline/configs"
    )
    params = parser.parse_args()

    config_path = Path("./configs") / params.config
    with open(config_path, 'r') as fs:
        config = yaml.safe_load(fs)
    return config_path, config
