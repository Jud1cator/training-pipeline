from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    @abstractmethod
    def get_value(self, *args, **kwargs):
        """
        Returns the value of the metric
        """
