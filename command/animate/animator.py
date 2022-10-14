from abc import ABC, abstractmethod
from typing import Callable, Generic, List, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from pde_solver.mesh import Interval

T = TypeVar("T", float, np.ndarray)


class NothingToPlotError(Exception):
    ...


class SolutionAnimator(ABC, Generic[T]):
    spatial_grid: np.ndarray
    _temporal_grid: np.ndarray = np.array([])

    _animation_available = False

    def set_grid(self, interval: Interval, mesh_size: int):
        self.spatial_grid = np.linspace(interval.a, interval.b, mesh_size)

    @property
    def temporal_grid(self) -> np.ndarray:
        return self._temporal_grid

    @temporal_grid.setter
    def temporal_grid(self, temporal_grid: np.ndarray):
        if len(self.temporal_grid) != 0 and len(temporal_grid) != len(
            self._temporal_grid
        ):
            raise ValueError("If frames is once set it can't be changed.")
        else:
            self._temporal_grid = temporal_grid

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
        self,
        spatial_grid: np.ndarray,
        temporal_grid: np.ndarray,
        function_values: np.ndarray,
        *label: str
    ):
        ...

    def show(self):
        if self._animation_available:
            self._setup()
            plt.show()
            plt.close()

        else:
            raise NothingToPlotError

    @abstractmethod
    def _setup(self):
        ...
