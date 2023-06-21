from configlate.registry import register_loss
import torch.nn as nn

@register_loss
def crossentropy():
    return nn.CrossEntropyLoss()