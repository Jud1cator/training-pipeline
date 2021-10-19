import pytorch_lightning as pl
import numpy as np

from metrics import ConfusionMatrix
from utils.visualization import visualize_batch
from utils.visualization import plot_confusion_matrix


class TestTask(pl.LightningModule):

    def __init__(
        self,
        datamodule,
        network,
        debug: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.classes = datamodule.classes
        self.num_classes = len(self.classes)

        self.net = network

        self.cm = ConfusionMatrix(self.num_classes)

        self.visualize_train = debug
        self.visualize_val = debug
        self.visualize_test = debug

    def forward(self, x):
        return self.net(x)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        img, true = batch
        if self.visualize_test:
            visualize_batch(img)
            self.visualize_test = False
        pred = np.argmax(self(img).cpu().numpy(), axis=1)
        self.cm.update(true.cpu().numpy(), pred)

    def test_epoch_end(self, outputs) -> None:
        cm = self.cm.get_value()
        precision = np.diag(cm) / cm.sum(axis=0)
        avg_precision = precision.sum() / len(precision)
        self.log('average_precision', avg_precision)
        plot_confusion_matrix(cm, categories=self.classes, sort=False)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.net.load_state_dict(state_dict)
