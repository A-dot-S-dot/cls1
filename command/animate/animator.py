from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar

import defaults
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from pde_solver.mesh import Interval

T = TypeVar("T", float, np.ndarray)


class NothingToAnimateError(Exception):
    ...


class SolutionAnimator(ABC, Generic[T]):
    interval: int
    spatial_grid: np.ndarray
    start_time: float
    save: Optional[str] = None
    frame_factor: float

    _temporal_grid: np.ndarray = np.array([])
    _animation: animation.FuncAnimation

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

    @property
    def start_index(self) -> int:
        return np.where(self.temporal_grid >= self.start_time)[0][0]

    @property
    def frames_per_second(self) -> int:
        return int(
            (len(self._temporal_grid) - self.start_index)
            / (self.temporal_grid[-1] - self.temporal_grid[self.start_index])
            / self.frame_factor
        )

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

            if self.save:
                self._animation.save(
                    self.save, writer="ffmpeg", fps=self.frames_per_second
                )
            else:
                plt.show()

            plt.close()
        else:
            raise NothingToAnimateError

    @abstractmethod
    def _setup(self):
        ...
