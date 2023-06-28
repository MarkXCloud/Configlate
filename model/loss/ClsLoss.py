from configlate import registry
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClsCrossEntropy(nn.Module):

    def __call__(self, *args, **kwargs) -> dict:
        return dict(loss=F.cross_entropy(*args, **kwargs))
    @staticmethod
    def post_process(pred:torch.tensor):
        return torch.argmax(pred, dim=-1)

@registry.loss
def crossentropy():
    return ClsCrossEntropy()

class ClsMSE(nn.Module):
    def __call__(self, *args, **kwargs) -> dict:
        return dict(loss=F.mse_loss(*args, **kwargs))
    @staticmethod
    def post_process(pred:torch.tensor):
        return torch.argmax(pred, dim=-1)

@registry.loss
def mse():
    return ClsMSE()