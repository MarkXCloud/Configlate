from configlate.registry import model,_build
from .loss import *
from .ResNet import *

def build_model(model_name,pretrained=False,**kwargs):
    base_model = _build(name=model_name,registry=model,**kwargs)
    if pretrained:
        base_model.load_state_dict(torch.load(pretrained,map_location="cpu"))
    return base_model