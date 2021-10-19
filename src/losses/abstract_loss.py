from abc import ABC, abstractmethod


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self, input, target):
        """
        Calculates and returns loss value
        """
