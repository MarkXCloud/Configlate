from configlate import registry
import torch

@registry.optimizer
def sgd(**kwargs):
    return torch.optim.SGD(**kwargs)