from typing import Callable, Any
import sys


class Registry(dict):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self._name = name

    def __call__(self, fn: Callable[..., Any], *args, **kwargs) -> Callable[..., Any]:

        # lookup containing module
        mod = sys.modules[fn.__module__]
        module_name_split = fn.__module__.split('.')
        module_name = module_name_split[-1] if len(module_name_split) else ''

        # add model to __all__ in module
        model_name = fn.__name__
        if hasattr(mod, '__all__'):
            mod.__all__.append(model_name)
        else:
            mod.__all__ = [model_name]  # type: ignore

        # add entries to registry dict/sets
        self[model_name] = fn

        return fn
