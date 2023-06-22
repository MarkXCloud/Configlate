from configlate import registry
import torch

@registry.optimizer
def adamw(**kwargs):
    return torch.optim.AdamW(**kwargs)