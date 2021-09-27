from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from argparse import ArgumentParser

from models.effnets import EfficientNetLite0
from tasks.test_task import TestTask
from data_modules import ClassificationDataModule
from utils.visualization import plot_confusion_matrix


def prepare_run(name, seed):
    torch.manual_seed(seed)
    all_runs_dir = Path("./runs")
    date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    run_name = str(date) + f'_{name}'
    run_dir = all_runs_dir / run_name
    test_res_dir = run_dir / 'results'
    test_res_dir.mkdir(exist_ok=True, parents=True)
    return run_dir, test_res_dir


def main(params):
    classes = ('Positive', 'Negative')

    dm = ClassificationDataModule(
        data_dir=params.data_path,
        input_shape=(227, 277),
        classes=classes,
        val_split=0.0,
        test_split=1.0,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=4,
    )

    network = EfficientNetLite0(input_shape=dm.input_shape, num_classes=len(classes))

    model = TestTask(dm, network, debug=True)
    model.load_state_dict(torch.load(params.weights_path))
    model.eval()

    trainer = Trainer()

    trainer.test(model, datamodule=dm)
    cm = model.cm.get_confusion_matrix()
    plot_confusion_matrix(cm, categories=classes, sort=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", required=True,
        help="Path to folder with test images"
    )
    parser.add_argument(
        "-w", "--weights-path", required=True,
        help="Path to .json file with model weights"
    )
    params = parser.parse_args()
    main(params)
