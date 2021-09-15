import numpy as np


class ConfusionMatrix:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self._cm = np.zeros((n_classes, n_classes))

    def update(self, target, prediction):
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                self._cm[i, j] += ((target == i) & (prediction == j)).sum()

    def get_confusion_matrix(self):
        return self._cm
