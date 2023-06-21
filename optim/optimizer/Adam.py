from configlate.registry import register_optimizer
import torch

@register_optimizer
def adam(**kwargs):
    return torch.optim.Adam(**kwargs)