from configlate import registry
import torch

@registry.scheduler
def steplr(*args, **kwargs):
    return torch.optim.lr_scheduler.StepLR(*args, **kwargs)

@registry.scheduler
def multisteplr(*args, **kwargs):
    return torch.optim.lr_scheduler.MultiStepLR(*args, **kwargs)
