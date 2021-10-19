import numpy as np

from metrics import AbstractMetric


class F1Score(AbstractMetric):

    def __init__(self):
        super().__init__()

    def get_value(self, confusion_matrix: np.ndarray):
        diag = np.diag(confusion_matrix)
        precision = np.mean(np.nan_to_num(diag / confusion_matrix.sum(axis=0)))
        recall = np.mean(np.nan_to_num(diag / confusion_matrix.sum(axis=1)))
        return 2 * precision * recall / (precision + recall)
