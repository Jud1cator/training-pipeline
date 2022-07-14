import inspect
import pkgutil

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import data_modules, losses, metrics, tasks
from losses.segmentation.combo_loss import ComboLoss
from losses.segmentation.dice_loss import DiceLoss
from metrics.iou_metric import IoUMetric

from models.segmentation.unet import UNet
from models.segmentation.Tiramisu.tiramisu import FCDenseNet

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
        'UNet': UNet,
        'FCDenseNet': FCDenseNet
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
