from configlate.registry import register_dataset,_build

__all__=['build_dataeset']

def build_dataeset(dataset_name,**kwargs):
    return _build(name=dataset_name,registry=register_dataset,**kwargs)