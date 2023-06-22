from configlate import registry
import torch.nn as nn

@registry.loss
def crossentropy():
    return nn.CrossEntropyLoss()