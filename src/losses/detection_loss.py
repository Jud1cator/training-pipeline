from effdet.loss import DetectionLoss
from omegaconf import OmegaConf


def create_detection_loss(
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 1.5,
        delta: float = 0.1,
        box_loss_weight: float = 50.0,
        label_smoothing: float = 0.,
        legacy_focal: bool = False,
        jit_loss: bool = False
):
    config = OmegaConf.create()
    config.num_classes = num_classes
    config.alpha = alpha
    config.gamma = gamma
    config.delta = delta
    config.box_loss_weight = box_loss_weight
    config.label_smoothing = label_smoothing
    config.legacy_focal = legacy_focal
    config.jit_loss = jit_loss
    return DetectionLoss(config)
