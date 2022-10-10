from typing import List
from abc import ABC, abstractmethod

import numpy as np


class DiscreteSolution:
    """Class representing discrete solution."""

    time: List[float]
    solution: List[np.ndarray]

    def __init__(self, start_time: float, initial_data: np.ndarray):
        """The first dimension should correspond to the DOFs."""
        self.time = [start_time]
        self.solution = [initial_data]

    @property
    def dimension(self):
        return len(self.solution[0])

    @property
    def initial_data(self) -> np.ndarray:
        return self.solution[0].copy()

    @property
    def start_time(self) -> float:
        return self.time[0]

    @property
    def end_solution(self) -> np.ndarray:
        return self.solution[-1].copy()

    @property
    def end_time(self) -> float:
        return self.time[-1]

    def add_solution(self, time_step: float, solution: np.ndarray):
        new_time = self.time[-1] + time_step
        self.time.append(new_time)
        self.solution.append(solution)
