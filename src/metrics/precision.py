import numpy as np

from metrics import AbstractMetric


class Recall(AbstractMetric):

    MODES = ['macro', 'micro']

    def __init__(self, mode='macro'):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode for Precision: {mode}")
        self.mode = mode

    def get_value(self, confusion_matrix: np.ndarray):
        if self.mode == 'macro':
            return np.mean(np.nan_to_num(np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)))
        elif self.mode == 'micro':
            return np.nan_to_num(np.trace(confusion_matrix) / confusion_matrix.sum())
