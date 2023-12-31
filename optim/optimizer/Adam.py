from configlate import registry
import torch

@registry.optimizer
def adam(**kwargs):
    return torch.optim.Adam(**kwargs)