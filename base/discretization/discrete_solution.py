from typing import List, Tuple

import numpy as np

from .abstract import SolverSpace, EmptySpace


class DiscreteSolution:
    """Class representing discrete solution."""

    time: np.ndarray
    time_steps: List[float]
    values: np.ndarray
    grid: np.ndarray
    space: SolverSpace

    def __init__(self, initial_data: np.ndarray, start_time=0.0, grid=None, space=None):
        """The first dimension should correspond to the number of time steps and
        the second to DOFs dimension.

        """
        self.time = np.array([start_time])
        self.time_steps = []
        self.grid = grid if grid is not None else np.empty(0)
        self.values = np.array([initial_data])
        self.space = space or EmptySpace()

    @property
    def dimension(self) -> float | Tuple[float, ...]:
        dimension = self.initial_data.shape
        if len(dimension) == 1:
            return dimension[0]
        else:
            return dimension

    @property
    def initial_data(self) -> np.ndarray:
        return self.values[0].copy()

    @property
    def end_values(self) -> np.ndarray:
        return self.values[-1].copy()

    @property
    def start_time(self) -> float:
        return self.time[0]

    @property
    def end_time(self) -> float:
        return self.time[-1]

    def add_solution(self, time_step: float, solution: np.ndarray):
        new_time = self.time[-1] + time_step

        self.time = np.append(self.time, new_time)
        self.time_steps.append(time_step)
        self.values = np.append(self.values, np.array([solution]), axis=0)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(grid={self.grid}, time={self.time}, values={self.values}, time_steps={self.time_steps})"
        )
