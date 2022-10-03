from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from mesh import Interval

import numpy as np

T = TypeVar("T", float, np.ndarray)


class NoExactSolutionError(Exception):
    ...


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
        raise NoExactSolutionError("No exact solution exist.")

    def exact_solution_at_end_time(self, x: float) -> T:
        return self.exact_solution(x, self.end_time)
