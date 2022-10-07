from abc import ABC, abstractmethod
import numpy as np


class SystemVector(ABC):
    @abstractmethod
    def __call__(self, *args) -> np.ndarray:
        ...
