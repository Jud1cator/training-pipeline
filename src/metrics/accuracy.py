import numpy as np

from metrics import AbstractMetric


class Accuracy(AbstractMetric):

    def __init__(self):
        super().__init__()

    def get_value(self, confusion_matrix: np.ndarray):
        return np.nan_to_num(np.trace(confusion_matrix) / confusion_matrix.sum())
