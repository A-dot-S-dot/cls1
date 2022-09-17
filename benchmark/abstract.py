from abc import ABC, abstractmethod
from mesh import Interval

import numpy as np


class Benchmark(ABC):
    _domain: Interval
    _T: float

    @property
    def domain(self) -> Interval:
        return self._domain

    @property
    def T(self) -> float:
        return self._T

    @abstractmethod
    def initial_data(self, x: float) -> float:
        ...

    def exact_solution(self, x: float, t: float) -> float:
        return np.nan

    def has_exact_solution(self) -> bool:
        return not np.isnan(self.exact_solution(0, 0))

    def exact_solution_at_T(self, x: float) -> float:
        return self.exact_solution(x, self.T)
