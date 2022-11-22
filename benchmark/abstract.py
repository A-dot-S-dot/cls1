from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from pde_solver.mesh import Interval

T = TypeVar("T", float, np.ndarray)


class NoExactSolutionError(Exception):
    ...


class Benchmark(ABC, Generic[T]):
    domain: Interval
    start_time: float
    end_time: float

    @abstractmethod
    def __init__(self, end_time=None, **benchmark_parameters):
        ...

    @abstractmethod
    def initial_data(self, x: float) -> T:
        ...

    def exact_solution(self, x: float, t: float) -> T:
        raise NoExactSolutionError("No exact solution exist.")

    def exact_solution_at_end_time(self, x: float) -> T:
        return self.exact_solution(x, self.end_time)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(domain={self.domain}, start_time={self.start_time}, end_time={self.end_time})"
        )
