from .misc import on_main_process
from abc import ABCMeta, abstractmethod
from tensorboardX import SummaryWriter
from pathlib import Path
from shutil import copy
import datetime
import wandb
from omegaconf import OmegaConf
from prettytable import PrettyTable
import json


class _BaseLogger(metaclass=ABCMeta):
    def __init__(self):
        ...

    @abstractmethod
    def init(self, *args, **kwargs):
        ...

    @abstractmethod
    def log(self, *args, **kwargs):
        ...

    @abstractmethod
    def save(self, *args, **kwargs):
        ...

    @abstractmethod
    def finish(self, *args, **kwargs):
        ...


class SysLogger(_BaseLogger):
    save_dir = None
    _table = PrettyTable()
    _table.float_format = ".6"
    _buffer = []

    @staticmethod
    def init(cfg, *args, **kwargs):
        SysLogger.save_dir = cfg.saver.save_dir

    @staticmethod
    def log(trace_log: dict, *args, **kwargs):
        SysLogger._buffer.append(trace_log)
        SysLogger._table.clear_rows()
        if not SysLogger._table.field_names:# fields haven't been assigned
            SysLogger._table.field_names = trace_log.keys()
        SysLogger._table.add_row(trace_log.values())

        print(SysLogger._table)

    def save(self, file, **kwargs):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        copy(src=file, dst=self.save_dir)

    def finish(self, *args, **kwargs):
        with open(SysLogger.save_dir / "result.json", "w") as f:
            json.dump(SysLogger._buffer, f)


class TensorboardLogger(_BaseLogger):
    def init(self, cfg, *args, **kwargs):
        self.writer = SummaryWriter(logdir=cfg.saver.save_dir)

    def log(self, trace_log: dict, step: int, **kwargs):
        for k, v, in trace_log.items():
            self.writer.add_scalar(k, v, step)

    def save(self, *args, **kwargs):
        ...

    def finish(self):
        self.writer.close()


class WandBLogger:
    def __init__(self):
        pass

    @staticmethod
    def init(cfg, *args, **kwargs):
        wandb.init(project=cfg.wandb_log_name, config=OmegaConf.to_container(cfg), *args, **kwargs)

    @staticmethod
    def log(trace_log, step, *args, **kwargs):
        wandb.log(trace_log, step=step, *args, **kwargs)

    @staticmethod
    def save(file, *args, **kwargs):
        wandb.save(file)

    @staticmethod
    def finish(*args, **kwargs):
        wandb.finish(*args, **kwargs)


class Logger(_BaseLogger):
    loggers = []

    @staticmethod
    @on_main_process
    def init(cfg, *args, **kwargs):
        cfg.saver.save_dir = Path(cfg.saver.save_dir) / Path(
            cfg.config).stem / datetime.datetime.today().strftime(
            '%Y%m%d_%H_%M_%S')

        Logger.loggers.append(SysLogger())
        if cfg.wandb:
            Logger.loggers.append(WandBLogger())
        if cfg.tb:
            Logger.loggers.append(TensorboardLogger())

        for logger in Logger.loggers:
            logger.init(cfg=cfg, *args, **kwargs)

    @staticmethod
    @on_main_process
    def log(*args, **kwargs):
        for logger in Logger.loggers:
            logger.log(*args, **kwargs)

    @staticmethod
    @on_main_process
    def save(*args, **kwargs):
        for logger in Logger.loggers:
            logger.save(*args, **kwargs)

    @staticmethod
    @on_main_process
    def finish(*args, **kwargs):
        for logger in Logger.loggers:
            logger.finish(*args, **kwargs)
