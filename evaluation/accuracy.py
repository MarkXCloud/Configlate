import torch
from configlate import registry
from configlate.util import all_gather
from ._BaseEvaluator import _Evaluator
from collections import OrderedDict


class Accuracy(_Evaluator):
    def __init__(self,topk=(1,)):
        super().__init__()
        self._buffer_pred = []
        self._buffer_label = []
        self.topk = topk

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

        maxk = max(self.topk)
        _, all_pred = all_pred.topk(maxk, 1, True, True)
        all_pred = all_pred.t()
        correct = all_pred.eq(all_label.view(1, -1).expand_as(all_pred))
        num = all_label.shape[0]
        res = OrderedDict()
        for k in self.topk:
            correct_k = correct[:k].float().sum()
            res.update({f"accuracy@top{k}": correct_k.div_(num).item()})

        self._buffer_pred = []
        self._buffer_label = []
        return res


@registry.metric
def accuracy(*args,**kwargs):
    return Accuracy(*args,**kwargs)