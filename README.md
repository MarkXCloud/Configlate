# A template codebase for various deep learning paradigms

This repo is a template codebase for various deep learning training paradigms, such as image classification, object
detection and so on. 

Based ont Template, I tried to put everything together with `.yaml` configs and the `build` function
**Configlate** can work with various functions necessary in a training paradigm:

1. Custom model definition.
2. Custom dataset definition,built by `build_dataset`, with [albumentations](https://albumentations.ai/docs/) for data augmentation.
3. Config file in `.yaml`,load everything from the corresponding build function.
4. Distributed training.
5. [WandB](https://wandb.ai/site) for logging the info of everything at first, and tracing the loss or other metrics curve.
6. Evaluation.Initiate the evaluator in config, call it in test loop to record predictions, and finally compute all the metrics. If you want to use a custom metric you should rewrite `add_batch()` method and `compute()`method.
7. Saver to save the latest model with custom interval and the best model with specific metric.
8. Model #params and #MACS supported by [torchinfo](https://github.com/TylerYep/torchinfo) and [ptflops](https://github.com/LukasHedegaard/ptflops).

# Requirements

- torch
- torchvision
- PyYAML
- tqdm
- albumentations
- torchinfo
- ptflops

# Training

To train a model directly:

```sh
python train.py configs/res50_cifar10.yaml
```

Then you can find your run under `runs/res50_cifar10/$local_time$/` with your config file and `.pt` weights.

To use distributed training:

```sh
torchrun --nnode 1 --nproc_per_node 8 train.py configs/res50_cifar10.yaml
```

# Calculate parameters and MACS

To show the #params and MACS of your model:

```sh
python model_info.py configs/res50_cifar10.yaml --input_size 1,3,224,224
```



# Custom modules

To be done.


# Acknowledgement
Thanks to every wonderful module used in this repo. Your effort helps me to finish all parts of the repo in an easier way.