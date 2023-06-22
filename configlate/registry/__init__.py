from ._registers import *

def _build(name:str,registry:Registry,**kwargs):
    if not name in registry:
        raise KeyError(f"{name} cannot be found. Please make sure you have a registry on it and import it in the __init__.py")
    fn = registry[name]
    return fn(**kwargs)
