import torch
import argparse
from torchinfo import summary
from configlate import load_yaml
import timm
from ptflops import get_model_complexity_info


if __name__=='__main__':
    parser = argparse.ArgumentParser("Basic necessary kwargs for calculating flops", add_help=False)
    parser.add_argument('config', type=str, default='', help="model.yaml path")
    parser.add_argument('--input_size', type=str, default="1,3,224,224", help="input shape")
    args = parser.parse_args()

    module_dict = load_yaml(args.config)
    module_dict['model'].pop('loss')
    model_params = module_dict['model']
    model = timm.create_model(**model_params)
    input_size =eval(args.input_size)
    assert isinstance(input_size,tuple),"--input_size must be a tuple"
    img_size = input_size[1:]
    del module_dict

    with torch.cuda.device(0):
        summary(model,input_size)
        macs, params = get_model_complexity_info(model, img_size, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))