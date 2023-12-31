from configlate.registry import metric, _build
from .accuracy import *


def build_metric(metric_name, **kwargs):
    return _build(name=metric_name, registry=metric, **kwargs)
