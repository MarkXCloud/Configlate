import torch
import argparse
from torchinfo import summary
from configlate import load_yaml_cfg
from model import build_model
from ptflops import get_model_complexity_info


if __name__=='__main__':
    parser = argparse.ArgumentParser("Basic necessary kwargs for calculating flops", add_help=False)
    parser.add_argument('config', type=str, default='', help="model.yaml path")
    parser.add_argument('--input-size', type=str, default="1,3,224,224", help="input shape")
    args = parser.parse_args()

    args = load_yaml_cfg(args)
    model = build_model(**args.model)
    input_size = tuple(int(i) for i in args.input_size.split(','))
    img_size = input_size[1:]

    with torch.cuda.device(0):
        summary(model,input_size)
        macs, params = get_model_complexity_info(model, img_size, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))