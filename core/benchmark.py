from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from .boundary import BOUNDARY_CONDITIONS
from .error import CustomError
from .mesh import Interval

T = TypeVar("T", float, np.ndarray)


class NoExactSolutionError(CustomError):
    ...


class Benchmark(ABC, Generic[T]):
    problem: str
    domain: Interval
    start_time = 0.0
    end_time: float
    boundary_conditions: BOUNDARY_CONDITIONS

    @abstractmethod
    def __init__(self, end_time=None, **benchmark_parameters):
        ...

    @abstractmethod
    def initial_data(self, x: float) -> T:
        ...

    def inflow_left(self, t: float) -> T:
        raise AttributeError("No left inflow speciefied.")

    def inflow_right(self, t: float) -> T:
        raise AttributeError("No right inflow speciefied.")

    def exact_solution(self, x: float, t: float) -> T:
        raise NoExactSolutionError("No exact solution exists.")

    def exact_solution_at_end_time(self, x: float) -> T:
        return self.exact_solution(x, self.end_time)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(domain={self.domain}, start_time={self.start_time}, end_time={self.end_time})"
        )
