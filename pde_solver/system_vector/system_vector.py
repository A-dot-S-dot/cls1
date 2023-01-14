from abc import ABC, abstractmethod

import numpy as np


class SystemVector(ABC):
    _assembled = False

    def assemble(self, *args):
        if not self._assembled:
            self._assemble(*args)
            self._assembled = True

    def _assemble(self, *args):
        ...

    @staticmethod
    def assemble_before_call(call_function):
        def new_call_function(cls, *args):
            cls.assemble(*args)
            value = call_function(cls, *args)

            cls._assembled = False

            return value

        return new_call_function

    @abstractmethod
    def __call__(self, *args) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__
