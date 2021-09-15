from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from geffnet import efficientnet_lite0


class Registry:
    MODELS = dict()

    TASKS = dict()

    DATAMODULES = dict()

    LOSSES = {
        'CrossEntropyLoss': CrossEntropyLoss
    }

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
        cls.DATAMODULES[datamodule_class.__name__] = datamodule_class
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
