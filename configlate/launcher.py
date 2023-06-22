import timm
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, reduce
from tqdm import tqdm
from os import makedirs, path as osp
import time
from .yaml_config import load_yaml
from dataset import build_dataeset
from model import build_loss
from optim import build_optimizer, build_scheduler
import evaluate
from paradigm import build_paradigm

__all__=['launch','Saver']

accelerator = Accelerator(log_with=['wandb'])


def launch(args):
    paradigm, model, loss, train_loader, test_loader, optimizer, scheduler, epoch, metric, saver = \
        prepare_everything(args)

    loss_dict = {}

    # start training
    for e in range(epoch):
        train_pbar = tqdm(train_loader, desc=f'Train epoch {e}', disable=not accelerator.is_local_main_process)

        model.train()
        for x, y in train_pbar:
            optimizer.zero_grad()
            loss_dict = paradigm.train(model, x, y, loss)
            accelerator.backward(loss_dict['loss'])
            optimizer.step()
        scheduler.step()

        train_loss_dict = reduce(loss_dict)  # reduce the loss among all devices
        accelerator.print(','.join([f'{k} = {v:.5f}' for k, v in train_loss_dict.items()]))

        test_pbar = tqdm(test_loader, desc=f'Test epoch {e}', disable=not accelerator.is_local_main_process)

        model.eval()
        for x, label in test_pbar:
            pred = paradigm.inference(model, x)
            metric.add_batch(references=label, predictions=pred)

        metrics = metric.compute()
        accelerator.print(','.join([f'{k} = {v:.5f}' for k, v in metrics.items()]))

        saver.save_latest_model(model)
        saver.save_best_model(model, metrics)

        trace_log = dict(
            **metrics,
            **train_loss_dict,
            learning_rate=optimizer.optimizer.param_groups[0]['lr'],
        )
        accelerator.log(trace_log, step=e)

    accelerator.end_training()


def prepare_everything(args):
    set_seed(args.seed)

    module_dict = load_yaml(args.config)

    # load everything from the module
    paradigm_params = module_dict['paradigm']
    loss_params = module_dict['model'].pop('loss')
    model_params = module_dict['model']
    dataset_params = module_dict['dataset']
    optimizer_params = module_dict['optimizer']
    scheduler_params = module_dict['scheduler']
    metric_params = module_dict['metric']
    saver_params = module_dict['saver']
    log_name = module_dict.pop('log_name')

    # basic info for the wandb log
    accelerator.init_trackers(log_name, config=module_dict)

    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True
    paradigm = build_paradigm(paradigm_params)
    model = timm.create_model(**model_params)
    loss = build_loss(**loss_params)
    train_loader, test_loader = build_dataeset(**dataset_params)
    optimizer = build_optimizer(params=model.parameters(),**optimizer_params)
    scheduler = build_scheduler(optimizer=optimizer,**scheduler_params)
    epoch = module_dict['epoch']
    metric = evaluate.load(metric_params)
    saver = build_saver(config_file=args.config,**saver_params)

    model, optimizer, train_loader, test_loader \
        = accelerator.prepare(model, optimizer, train_loader, test_loader)

    return paradigm, model, loss, train_loader, test_loader, optimizer, scheduler, epoch, metric, saver


class Saver:
    """
    Saver acts as a scheduler to save the latest model and the best model.
    """

    def __init__(self, save_interval: int, higher_is_better: bool, config_file: str, monitor: str,
                 root: str = './runs/'):
        """
        :param save_interval: when we want to save the latest model, it saves it per $save_step$ epochs.
        :param higher_is_better: when we want to save the best model, we should point out what is 'best', higher_is_better means\
        if the metric we choose is higher, then we get a better model, so we save it!
        :param monitor: the metric that we want to observe for best model, e.g., accuracy
        """
        # create save dir
        save_dir = osp.join(root, osp.basename(config_file).split('.')[0],
                            time.strftime('%Y%m%d_%H_%M_%S', time.localtime(time.time())))
        if accelerator.is_local_main_process:
            import shutil
            makedirs(save_dir)
            shutil.copy(src=config_file, dst=save_dir)

        self.save_dir = save_dir
        self._save_interval = save_interval
        # count for epochs, when the count meets save_interval, it saves the latest model
        self._cnt = 1

        self.hib = higher_is_better
        self._metric = -1 if higher_is_better else 65535
        self.monitor = monitor

    def save_latest_model(self, model):
        if self._cnt == self._save_interval:
            accelerator.save(accelerator.get_state_dict(model), f=osp.join(self.save_dir, "latest.pt"))
            self._cnt = 1
            accelerator.print(f"Save latest model under {self.save_dir}")
        else:
            self._cnt += 1

    def save_best_model(self, model, metric):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            accelerator.save(accelerator.get_state_dict(model), f=osp.join(self.save_dir, "best.pt"))
            self._metric = metric
            accelerator.print(f"Save new best model under {self.save_dir}")


def build_saver(**kwargs):
    return Saver(**kwargs)
