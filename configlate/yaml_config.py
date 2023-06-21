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


def fuse_yaml_cfg(raw_yaml_dict: OrderedDict):
    assert '__base__' in raw_yaml_dict, f"__base__ key missing in current yaml file"
    base_cfg = raw_yaml_dict.pop('__base__')

    if raw_yaml_dict['dataset'].get('img_size', False):
        raw_yaml_dict['dataset']['img_size'] = eval(raw_yaml_dict['dataset']['img_size'])[-2:]
    assert isinstance(raw_yaml_dict['dataset']['img_size'], tuple), "img_size must bu a tuple!"

    raw_yaml_dict['model'].update({'num_classes': raw_yaml_dict['dataset']['num_classes']})
    raw_yaml_dict.update({'saver': base_cfg['saver']})

    info = dict(
        log_name=base_cfg['wandb_log_name'],
        epoch=base_cfg['epoch'],
        model=raw_yaml_dict['model'],
        optimizer=raw_yaml_dict['optimizer'],
        scheduler=raw_yaml_dict['scheduler'],
        dataset=raw_yaml_dict['dataset'],
        save_dir=base_cfg['saver'],
    )

    raw_yaml_dict.update({'info': info})

    return raw_yaml_dict


def load_yaml(file_name: str):
    assert file_name.endswith('.yaml'), f"{file_name} is not a .yaml file"
    return fuse_yaml_cfg(load_yaml_cfg(file_name))


if __name__ == '__main__':
    c = load_yaml_cfg('../configs/res50_cifar10.yaml')
    c = fuse_yaml_cfg(c)
    print(c)
