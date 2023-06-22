from configlate.registry import optimizer,_build

__all__=['build_optimizer']

def build_optimizer(optimizer_name,**kwargs):
    return _build(name=optimizer_name,registry=optimizer,**kwargs)