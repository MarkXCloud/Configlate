import torch
from configlate import registry
from configlate.util import all_gather
from ._BaseEvaluator import _Evaluator


class Accuracy(_Evaluator):
    def __init__(self):
        super().__init__()
        self._buffer_pred = []
        self._buffer_label = []

    def add_batch(self, pred, label):
        self._buffer_pred.append(pred.cpu())
        self._buffer_label.append(label.cpu())
    @torch.no_grad()
    def compute(self):
        pred = torch.cat(self._buffer_pred, dim=0)
        label = torch.cat(self._buffer_label, dim=0)
        # gather across gpus
        all_pred = torch.cat(all_gather(pred),dim=0)
        all_label = torch.cat(all_gather(label),dim=0)
        num_correct = all_pred.eq_(all_label).float().sum().item()
        num = all_label.shape[0]
        self.buffer_pred = []
        self.buffer_label = []
        return dict(accuracy=num_correct / num)


@registry.metric
def accuracy():
    return Accuracy()