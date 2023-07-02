from configlate.registry import loss, _build
from .ClsLoss import *


def build_loss(loss_name, **kwargs):
    return _build(name=loss_name, registry=loss, **kwargs)
