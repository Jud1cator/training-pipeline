from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pkgutil
import inspect

import data_modules
import losses
import metrics
import models
import tasks


class Registry:
    DATA_MODULES = dict()
    LOSSES = {
        'CrossEntropyLoss': CrossEntropyLoss
    }
    METRICS = dict()
    MODELS = dict()
    TASKS = dict()
    OPTIMIZERS = {
        'Adam': Adam
    }
    SCHEDULERS = {
        'ReduceLROnPlateau': ReduceLROnPlateau
    }
    CALLBACKS = {
        'ModelCheckpoint': ModelCheckpoint,
        'EarlyStopping': EarlyStopping
    }

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
