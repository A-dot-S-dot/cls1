from typing import List

import numpy as np


class DiscreteSolution:
    """Class representing discrete solution."""

    time: List[float]
    values: np.ndarray

    def __init__(self, start_time: float, initial_data: np.ndarray):
        """The first dimension should correspond to the number of time steps and
        the second to DOFs dimension.

        """
        self.time = [start_time]
        self.values = np.array([initial_data])

    @property
    def dimension(self):
        return self.values.shape[1]

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
        self.time.append(new_time)
        self.values = np.append(self.values, np.array([solution]), axis=0)

    def __repr__(self) -> str:
        return str(self.values)
