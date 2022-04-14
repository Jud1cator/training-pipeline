from src.metrics.abstract_metric import AbstractMetric


class IoUMetric(AbstractMetric):
    def __init__(self, num_classes, smooth=1e-7):
        self.num_classes = num_classes
        self.smooth = smooth

        self.intersection = 0
        self.union = 0

    def update(self, prediction, target):
        for i in range(self.num_classes):
            ith_prediction = (prediction == i)
            ith_target = (target == i)
            intersection = (ith_prediction & ith_target).sum()
            union = ith_prediction.sum() + ith_target.sum() - intersection
            self.intersection += intersection
            self.union += union

    def reset(self):
        self.intersection = 0
        self.union = 0

    def get_value(self, *args, **kwargs):
        return (self.intersection + self.smooth) / (self.union + self.smooth)
