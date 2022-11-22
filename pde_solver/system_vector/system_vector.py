from abc import ABC, abstractmethod

import numpy as np


class SystemVector(ABC):
    @abstractmethod
    def __call__(self, *args) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__
