from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pkgutil
import inspect

import data_modules
import losses
import metrics
import models
import tasks


class Registry:
    DATA_MODULES = {}
    LOSSES = {
        'CrossEntropyLoss': CrossEntropyLoss
    }
    METRICS = {}
    MODELS = {}
    TASKS = {}
    TRANSFORMS = {
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
        'ReduceLROnPlateau': ReduceLROnPlateau
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
        cls.register_module(models, cls.MODELS)
        cls.register_module(tasks, cls.TASKS)
