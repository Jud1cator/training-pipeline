from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes + 1})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)