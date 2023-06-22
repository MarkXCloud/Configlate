from configlate import registry
from typing import List
import torch

class WarmUpMultiStep(torch.optim.lr_scheduler.ChainedScheduler):
    def __init__(self,optimizer,start_factor:float,warmup_iter:int,step_milestones:List[int],gamma:float):
        super().__init__(
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_iter),
                torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_milestones, gamma=gamma)],
        )

@registry.scheduler
def warmupmultistep(**kwargs):
    return WarmUpMultiStep(**kwargs)