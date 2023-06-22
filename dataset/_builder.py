from configlate.registry import dataset,_build

__all__=['build_dataeset']

def build_dataeset(dataset_name,**kwargs):
    return _build(name=dataset_name,registry=dataset,**kwargs)