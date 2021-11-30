import numpy as np

from metrics import AbstractMetric


class F1Score(AbstractMetric):

    MODES = ['macro', 'micro']

    def __init__(self, mode='macro'):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode for F1 Score: {mode}")
        self.mode = mode

    def get_value(self, confusion_matrix: np.ndarray):
        if self.mode == 'macro':
            diag = np.diag(confusion_matrix)
            precision = np.mean(np.nan_to_num(diag / confusion_matrix.sum(axis=0)))
            recall = np.mean(np.nan_to_num(diag / confusion_matrix.sum(axis=1)))
            return 2 * precision * recall / (precision + recall)
        elif self.mode == 'micro':
            # This is the same as accuracy
            return np.nan_to_num(np.trace(confusion_matrix) / confusion_matrix.sum())
