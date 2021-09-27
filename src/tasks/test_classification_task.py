import pytorch_lightning as pl
import numpy as np

from registry import Registry
from metrics import ConfusionMatrix
from utils.visualization import visualize_batch, plot_confusion_matrix


class TestClassificationTask(pl.LightningModule):

    def __init__(
            self,
            datamodule,
            network,
            debug=False,
            res_dir=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.classes = datamodule.classes
        self.num_classes = len(self.classes)

        self.net = Registry.MODELS[network['name']](
            num_classes=self.num_classes,
            input_shape=datamodule.input_shape,
            **network['params']
        ).get_model()

        self.cm = ConfusionMatrix(self.num_classes)

        self.visualize_train = debug
        self.visualize_val = debug
        self.visualize_test = debug

        self.res_dir = res_dir

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
        precision = self.cm.get_precision()
        recall = self.cm.get_recall()
        avg_precision = precision.sum() / len(precision)
        avg_recall = recall.sum() / len(recall)
        self.log('average_precision', avg_precision)
        self.log('average_recall', avg_recall)
        plot_confusion_matrix(
            self.cm.get_confusion_matrix(),
            path=self.res_dir/'confusion_matrix.png',
            categories=self.classes,
            sort=False
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        self.net.load_state_dict(state_dict, )
