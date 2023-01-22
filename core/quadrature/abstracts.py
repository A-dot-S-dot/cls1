"""This module contains abstract classes."""
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
from core.mesh import Interval

ScalarFunction = Callable[[float], float]


class Quadrature(ABC):
    _domain: Interval
    _nodes: List[float]
    _weights: np.ndarray

    @property
    def domain(self):
        return self._domain

    @property
    def nodes(self) -> List[float]:
        return self._nodes

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def degree(self) -> int:
        return len(self._nodes)

    @abstractmethod
    def integrate(self, f: ScalarFunction):
        ...
