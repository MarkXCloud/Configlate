from configlate.registry import register_scheduler,_build

__all__=['build_scheduler']

def build_scheduler(scheduler_name,**kwargs):
    return _build(name=scheduler_name,registry=register_scheduler,**kwargs)