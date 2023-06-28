from configlate.registry import metric,_build

__all__=['build_metric']

def build_metric(metric_name,**kwargs):
    return _build(name=metric_name,registry=metric,**kwargs)