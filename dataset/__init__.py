from configlate.registry import dataset, _build
from .CIFAR10 import *


def build_dataset(dataset_name, **kwargs):
    return _build(name=dataset_name, registry=dataset, **kwargs)
