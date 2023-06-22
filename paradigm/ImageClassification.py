from configlate import registry
from ._BaseParadigm import _Paradigm
import torch

class ImageClassificationParadigm(_Paradigm):
    @staticmethod
    def train(model,x,y,loss_fn):
        return dict(loss=loss_fn(model(x), y))
    @staticmethod
    @torch.no_grad()
    def inference(model, x):
        return torch.argmax(model(x), dim=-1)

@registry.paradigm
def imageclassification():
    return ImageClassificationParadigm()