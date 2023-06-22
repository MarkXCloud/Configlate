from configlate.registry import paradigm,_build

__all__=['build_paradigm']

def build_paradigm(paradigm_name,**kwargs):
    return _build(name=paradigm_name,registry=paradigm,**kwargs)