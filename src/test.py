import torch
from pytorch_lightning import Trainer
from argparse import ArgumentParser

from src.tasks import TestTask
from src.data_modules import ClassificationDataModule
from src.utils.visualization import plot_confusion_matrix
from src.models import SimpleNet


def main(params):
    classes = ('c1', 'c2', 'c3', 'c4', 'c5', 'c6')

    dm = ClassificationDataModule(
        data_dir=params.data_path,
        input_shape=(32, 32),
        classes=classes,
        val_split=0.0,
        test_split=1.0,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=4,
    )
    network = SimpleNet(input_shape=dm.input_shape, num_classes=len(classes))

    model = TestTask(dm, network, debug=False)
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
