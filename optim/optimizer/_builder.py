from configlate.registry import register_optimizer,_build

__all__=['build_optimizer']

def build_optimizer(optimizer_name,**kwargs):
    return _build(name=optimizer_name,registry=register_optimizer,**kwargs)