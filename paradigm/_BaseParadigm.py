from abc import ABCMeta,abstractmethod

class _Paradigm(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def train():
        ...

    @staticmethod
    @abstractmethod
    def inference():
        ...