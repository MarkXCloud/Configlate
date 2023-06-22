from configlate.registry import scheduler,_build

__all__=['build_scheduler']

def build_scheduler(scheduler_name,**kwargs):
    return _build(name=scheduler_name,registry=scheduler,**kwargs)