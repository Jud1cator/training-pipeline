from pydantic import BaseModel


class Config(BaseModel):
    name: str
    params: dict = {}


DEFAULT_CLASSIFICATION_LOSS = {
    'name': 'CrossEntropyLoss'
}

DEFAULT_SEGMENTATION_LOSS = {
    'name': 'ComboLoss'
}


DEFAULT_OPTIMIZER = {
    'name': 'Adam',
    'params': {
        'lr': 0.001
    }
}
