import numpy as np

from src.metrics.abstract_metric import AbstractMetric


class ConfusionMatrix(AbstractMetric):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self._cm = np.zeros((self.n_classes, self.n_classes))

    def update(self, input, target):
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                self._cm[i, j] += ((target == i) & (input == j)).sum()

    def reset(self):
        self._cm = np.zeros((self.n_classes, self.n_classes))

    def get_value(self):
        return self._cm
