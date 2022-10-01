from abc import ABC, abstractmethod
from argparse import Namespace


class Command(ABC):
    _args: Namespace

    def __init__(self, args: Namespace):
        self._args = args

    @abstractmethod
    def execute(self):
        ...
