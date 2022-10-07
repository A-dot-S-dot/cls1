from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from pde_solver.mesh import Interval

T = TypeVar("T", float, np.ndarray)


class NothingToPlotError(Exception):
    ...


class SolutionPlotter(ABC, Generic[T]):
    grid: np.ndarray

    _plot_available = False

    def set_grid(self, interval: Interval, mesh_size: int):
        self.grid = np.linspace(interval.a, interval.b, mesh_size)

    @abstractmethod
    def set_suptitle(self, suptitle: str):
        ...

    @abstractmethod
    def add_initial_data(self):
        ...

    @abstractmethod
    def add_exact_solution(self):
        ...

    @abstractmethod
    def add_function(self, function: Callable[[float], float], *label: str):
        ...

    @abstractmethod
    def add_function_values(
        self, grid: np.ndarray, function_values: np.ndarray, *label: str
    ):
        ...

    def show(self):
        if self._plot_available:
            self._setup()
            plt.show()
            plt.close()

        else:
            raise NothingToPlotError

    @abstractmethod
    def _setup(self):
        ...
