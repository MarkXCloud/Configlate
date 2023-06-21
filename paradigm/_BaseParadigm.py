from abc import ABCMeta,abstractmethod

class _Paradigm(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def train(**kwargs):
        ...

    @staticmethod
    @abstractmethod
    def inference(**kwargs):
        ...