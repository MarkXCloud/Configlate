import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
import configlate.util as util
from configlate.util import Logger
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import OrderedDict

from dataset import build_dataset
from model import build_model, build_loss
from optim import build_optimizer, build_scheduler
from evaluation import build_metric


def launch(args):
    args = load_yaml_cfg(args)

    util.init_distributed_mode(args)

    # seed all
    util.random_seed(seed=args.seed, rank=args.rank)

    # basic info for the log
    Logger.init(cfg=args)
    Logger.save(file=args.config)

    model = build_model(**args.model)
    loss_fn = build_loss(**args.loss)
    train_set, test_set = build_dataset(**args.dataset)
    optimizer = build_optimizer(params=model.parameters(), **args.optimizer)
    iter_scheduler = build_scheduler(optimizer=optimizer, **args.iter_scheduler)
    epoch_scheduler = build_scheduler(optimizer=optimizer, **args.epoch_scheduler)
    epoch = args.epoch
    metric = build_metric(**args.metric)
    saver = util.Saver(**args.saver)

    torch.cuda.empty_cache()

    device = torch.device(f'cuda:{args.device}')
    model = model.to(device)
    if args.compile:
        print("compiling model...")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    torch.backends.cudnn.benchmark = True

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device])
        sampler_train = DistributedSampler(train_set)
        sampler_val = DistributedSampler(test_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(test_set)

    batch_sampler_train = BatchSampler(
        sampler_train, batch_size=args.batch_size, drop_last=True)

    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers)



    loss_dict = {}

    # start training
    for e in range(epoch):
        if args.distributed:
            sampler_train.set_epoch(e)
        train_pbar = tqdm(train_loader, desc=f'Train epoch {e}', disable=not util.is_main_process())

        model.train()
        loss_fn.train()
        for x, y in train_pbar:
            x = x.to(args.device)
            y = y.to(args.device)

            outputs = model(x)
            loss_dict = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            iter_scheduler.step()
        epoch_scheduler.step()

        train_loss_dict = util.reduce_dict(loss_dict)  # reduce the loss among all devices
        train_loss_dict = {k: v.cpu().item() for k, v in train_loss_dict.items()}

        test_pbar = tqdm(test_loader, desc=f'Test epoch {e}', disable=not util.is_main_process())

        model.eval()
        loss_fn.eval()
        with torch.no_grad():
            for x, label in test_pbar:
                outputs = model(x)
                metric.add_batch(pred=outputs, label=label)

        metrics = metric.compute()

        trace_log = OrderedDict(
            **metrics,
            **train_loss_dict,
            learning_rate=optimizer.param_groups[0]['lr'],
        )
        Logger.log(trace_log, step=e)

        saver.save_latest_model(model)
        saver.save_best_model(model, metrics)

    Logger.finish()
    torch.distributed.destroy_process_group()


def launch_val(args):
    args = load_yaml_cfg(args)
    util.init_distributed_mode(args)
    # seed all
    util.random_seed(seed=args.seed, rank=args.rank)
    # basic info for the log
    args.wandb, args.tb = False, False  # we don't use wandb or tensorboard during validation
    Logger.init(cfg=args)
    Logger.save(file=args.config)

    _, test_set = build_dataset(**args.dataset)

    assert len(test_set) % (args.batch_size * util.get_world_size()) == 0, \
        f"test_set with length {len(test_set)} cannot div " \
        f"batch-size({args.batch_size}) * num_gpu({util.get_world_size()}), which will lead to " \
        f"result in-accurate due to the padding on last batch to align batch-size. Please assign " \
        f"a proper batch-size to make test_set could div."

    model = build_model(**args.model)

    torch.cuda.empty_cache()
    print("loading state_dict...")
    model.load_state_dict(torch.load(args.load_from, map_location="cpu"))
    device = torch.device(f'cuda:{args.device}')
    model = model.to(device)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device])
        sampler_val = DistributedSampler(test_set, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_set)

    test_loader = DataLoader(test_set, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.compile:
        print("compiling model...")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model,mode="max-autotune")

    metric = build_metric(**args.metric)

    test_pbar = tqdm(test_loader, desc=f'Testing', disable=not util.is_main_process())

    model.eval()
    with torch.no_grad():
        for x, label in test_pbar:
            outputs = model(x)
            metric.add_batch(pred=outputs, label=label)

    metrics = metric.compute()

    Logger.log(metrics)
    Logger.finish()
    torch.distributed.destroy_process_group()


def load_yaml_cfg(args):
    conf = OmegaConf.load(args.config)
    sys_args = OmegaConf.create(vars(args))
    conf = OmegaConf.merge(conf, sys_args)
    OmegaConf.resolve(conf)
    return conf
