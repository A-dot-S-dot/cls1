from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SystemFlux(ABC):
    @abstractmethod
    def __call__(self, *args) -> Tuple[np.ndarray, np.ndarray]:
        ...
