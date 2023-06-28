import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
import configlate.util as util
from configlate.util import WandBLogger
from tqdm import tqdm
from pathlib import Path
import datetime
from omegaconf import OmegaConf
from dataset import build_dataeset
from model import build_loss
from optim import build_optimizer, build_scheduler
from evaluation import build_metric


def launch(args):
    args = load_yaml_cfg(args)
    model, loss_fn, train_loader, test_loader, sampler_train, optimizer, scheduler, epoch, metric, saver = \
        prepare_everything(args)

    loss_dict = {}

    # start training
    for e in range(epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_pbar = tqdm(train_loader, desc=f'Train epoch {e}', disable=not util.is_main_process())

        model.train()
        loss_fn.train()
        for x, y in train_pbar:
            x = x.to(args.gpu)
            y = y.to(args.gpu)

            outputs = model(x)
            loss_dict = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
        scheduler.step()

        train_loss_dict = util.reduce_dict(loss_dict)  # reduce the loss among all devices
        print(','.join([f'{k} = {v:.5f}' for k, v in train_loss_dict.items()]))

        test_pbar = tqdm(test_loader, desc=f'Test epoch {e}', disable=not util.is_main_process())

        model.eval()
        loss_fn.eval()
        with torch.no_grad():
            for x, label in test_pbar:
                outputs = model(x)
                pred = loss_fn.post_process(outputs)
                metric.add_batch(pred=pred, label=label)

        metrics = metric.compute()
        print(','.join([f'{k} = {v:.5f}' for k, v in metrics.items()]))

        saver.save_latest_model(model)
        saver.save_best_model(model, metrics)

        trace_log = dict(
            **metrics,
            **train_loss_dict,
            learning_rate=optimizer.param_groups[0]['lr'],
        )
        WandBLogger.log(trace_log, step=e)

    WandBLogger.finish()
    torch.distributed.destroy_process_group()

def load_yaml_cfg(args):
    conf = OmegaConf.load(args.config)
    sys_args = OmegaConf.create(vars(args))
    conf = OmegaConf.merge(conf,sys_args)
    OmegaConf.resolve(conf)
    return conf

def prepare_everything(args):
    util.init_distributed_mode(args)
    # seed all
    util.random_seed(seed=args.seed, rank=args.rank)

    config_file = Path(args.config)
    args.saver.save_dir = Path(args.saver.save_dir) / config_file.stem / datetime.datetime.today().strftime(
        '%Y%m%d_%H_%M_%S')

    # basic info for the wandb log
    WandBLogger.init(project=args.log_name, config=OmegaConf.to_container(args))
    WandBLogger.save(args.config)
    if util.is_main_process():
        args.saver.save_dir.mkdir(parents=True, exist_ok=True)
        from shutil import copy
        copy(src=args.config, dst=args.saver.save_dir)

    model = timm.create_model(**args.model)
    loss_fn = build_loss(**args.loss)
    train_set, test_set = build_dataeset(**args.dataset)
    optimizer = build_optimizer(params=model.parameters(), **args.optimizer)
    scheduler = build_scheduler(optimizer=optimizer, **args.scheduler)
    epoch = args.epoch
    metric = build_metric(**args.metric)
    saver = util.Saver(**args.saver)

    torch.cuda.empty_cache()

    device = torch.device(f'cuda:{args.gpu}')
    model = model.to(device)

    torch.backends.cudnn.benchmark = True

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        sampler_train = DistributedSampler(train_set)
        sampler_val = DistributedSampler(test_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(test_set)

    batch_sampler_train = BatchSampler(
        sampler_train, batch_size=args.batch_size, drop_last=True)

    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers)

    return model, loss_fn, train_loader, test_loader, sampler_train, optimizer, scheduler, epoch, metric, saver
