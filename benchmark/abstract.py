from abc import ABC, abstractmethod
from mesh import Interval

import numpy as np


class Benchmark(ABC):
    end_time: float

    _domain: Interval

    @property
    def domain(self) -> Interval:
        return self._domain

    @abstractmethod
    def initial_data(self, x: float) -> float:
        ...

    def exact_solution(self, x: float, t: float) -> float:
        return np.nan

    def has_exact_solution(self) -> bool:
        return not np.isnan(self.exact_solution(0, 0))

    def exact_solution_at_end_time(self, x: float) -> float:
        return self.exact_solution(x, self.end_time)
