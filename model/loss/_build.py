from configlate.registry import loss,_build

__all__=['build_loss']

def build_loss(loss_name,**kwargs):
    return _build(name=loss_name,registry=loss,**kwargs)