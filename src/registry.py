import inspect
import pkgutil

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src import data_modules, losses, metrics, tasks
from src.losses.segmentation.combo_loss import ComboLoss
from src.losses.segmentation.dice_loss import DiceLoss
from src.metrics.iou_metric import IoUMetric
from src.models.classification.bit_vehicle_classifier_net import BITVehicleClassifierNet
from src.models.classification.effnets import EfficientNetB0, EfficientNetLite0
from src.models.classification.simple_net import SimpleNet
from src.models.detection.effdets import EfficientDet
from src.models.detection.mobiledets import SSDMobileDetCPU, SSDMobileNetV2MNASFPN
from src.models.segmentation.unet import UNet


class Registry:
    DATA_MODULES = {}
    LOSSES = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'DiceLoss': DiceLoss,
        'ComboLoss': ComboLoss
    }
    METRICS = {
        'IoU': IoUMetric
    }
    MODELS = {
        'BITVehicleClassifierNet': BITVehicleClassifierNet,
        'EfficientNetB0': EfficientNetB0,
        'EfficientNetLite0': EfficientNetLite0,
        'EfficientDet': EfficientDet,
        'SimpleNet': SimpleNet,
        'SSDMobileDetCPU': SSDMobileDetCPU,
        'SSDMobileNetV2MNASFPN': SSDMobileNetV2MNASFPN,
        'UNet': UNet
    }
    TASKS = {}
    TRANSFORMS = {
        'ToFloat': A.ToFloat,
        'Normalize': A.Normalize,
        'Resize': A.Resize,
        'SmallestMaxSize': A.SmallestMaxSize,
        'LongestMaxSize': A.LongestMaxSize,
        'PadIfNeeded': A.PadIfNeeded,
        'CenterCrop': A.CenterCrop,
        'RandomCrop': A.RandomCrop,
        'HorizontalFlip': A.HorizontalFlip,
        'RandomBrightnessContrast': A.RandomBrightnessContrast,
        'ToTensor': ToTensorV2
    }
    OPTIMIZERS = {
        'SGD': SGD,
        'Adam': Adam
    }
    SCHEDULERS = {
        'ReduceLROnPlateau': ReduceLROnPlateau,
        'CosineAnnealingLR': CosineAnnealingLR
    }
    CALLBACKS = {
        'ModelCheckpoint': ModelCheckpoint,
        'EarlyStopping': EarlyStopping
    }

    @classmethod
    def register_model(cls, model_class):
        cls.MODELS[model_class.__name__] = model_class
        return model_class

    @classmethod
    def register_task(cls, task_class):
        cls.TASKS[task_class.__name__] = task_class
        return task_class

    @classmethod
    def register_datamodule(cls, datamodule_class):
        cls.DATA_MODULES[datamodule_class.__name__] = datamodule_class
        return datamodule_class

    @classmethod
    def register_loss(cls, loss_class):
        cls.LOSSES[loss_class.__name__] = loss_class
        return loss_class

    @classmethod
    def register_optimizer(cls, optimizer_class):
        cls.OPTIMIZERS[optimizer_class.__name__] = optimizer_class
        return optimizer_class

    @classmethod
    def register_scheduler(cls, scheduler_class):
        cls.SCHEDULERS[scheduler_class.__name__] = scheduler_class
        return scheduler_class

    @classmethod
    def register_callback(cls, callback_class):
        cls.CALLBACKS[callback_class.__name__] = callback_class
        return callback_class

    @staticmethod
    def register_module(module, container):
        for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
            sub_module = importer.find_module(modname).load_module(modname)
            for name, obj in inspect.getmembers(sub_module):
                if inspect.isclass(obj):
                    container[name] = obj

    @classmethod
    def init_modules(cls):
        cls.register_module(data_modules, cls.DATA_MODULES)
        cls.register_module(losses, cls.LOSSES)
        cls.register_module(metrics, cls.METRICS)
        cls.register_module(tasks, cls.TASKS)
