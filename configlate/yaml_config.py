import yaml
import os
from typing import OrderedDict


def load_yaml_cfg(script_path) -> OrderedDict:
    with open(script_path, 'r') as f:
        d = yaml.safe_load(f.read())

    dir_name = os.path.dirname(script_path)

    for k, v in d.items():
        if isinstance(v, str) and v.endswith('.yaml'):
            d[k] = load_yaml_cfg(os.path.join(dir_name, k, v))
    return d


def fuse_yaml_cfg(raw_yaml_dict: OrderedDict)->OrderedDict:
    assert '__base__' in raw_yaml_dict, f"__base__ key missing in current yaml file"
    base_cfg = raw_yaml_dict.pop('__base__')
    base_cfg.update(raw_yaml_dict)

    if base_cfg['dataset'].get('img_size', False):
        base_cfg['dataset']['img_size'] = eval(base_cfg['dataset']['img_size'])[-2:]
        assert isinstance(base_cfg['dataset']['img_size'], tuple), "img_size must bu a tuple!"

    base_cfg['model'].update({'num_classes': base_cfg['dataset']['num_classes']})


    return base_cfg


def load_yaml(file_name: str):
    assert file_name.endswith('.yaml'), f"{file_name} is not a .yaml file"
    return fuse_yaml_cfg(load_yaml_cfg(file_name))


if __name__ == '__main__':
    c = load_yaml_cfg('../configs/res50_cifar10.yaml')
    c = fuse_yaml_cfg(c)
    print(c)
