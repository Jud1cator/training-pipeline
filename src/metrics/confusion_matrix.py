import numpy as np


class ConfusionMatrix:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self._cm = np.zeros((self.n_classes, self.n_classes))

    def update(self, target, prediction):
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                self._cm[i, j] += ((target == i) & (prediction == j)).sum()

    def reset(self):
        self._cm = np.zeros((self.n_classes, self.n_classes))

    def get_confusion_matrix(self):
        return self._cm

    def get_precision(self):
        return np.nan_to_num(np.diag(self._cm) / self._cm.sum(axis=0))

    def get_recall(self):
        return np.nan_to_num(np.diag(self._cm) / self._cm.sum(axis=1))
