from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from mesh import Interval

import numpy as np

T = TypeVar("T", float, np.ndarray)


class Benchmark(ABC, Generic[T]):
    start_time: float
    end_time: float

    _domain: Interval

    @property
    def domain(self) -> Interval:
        return self._domain

    @abstractmethod
    def initial_data(self, x: float) -> T:
        ...

    def exact_solution(self, x: float, t: float) -> T:
        raise NotImplementedError

    def has_exact_solution(self) -> bool:
        return not np.isnan(self.exact_solution(0, 0))

    def exact_solution_at_end_time(self, x: float) -> T:
        return self.exact_solution(x, self.end_time)
