from configlate import registry
import torch

@registry.scheduler
def linearlr(*args, **kwargs):
    return torch.optim.lr_scheduler.LinearLR(*args, **kwargs)
