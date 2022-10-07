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


class DiscreteSolutionObserver(ABC):
    @abstractmethod
    def update(self):
        ...


class DiscreteSolutionObservable(DiscreteSolution):
    """Every time discrete solutiion is changed it informes observers."""

    _observers: List[DiscreteSolutionObserver]

    def __init__(self, start_time: float, initial_data: np.ndarray):
        super().__init__(start_time, initial_data)
        self._observers = []

    def register_observer(self, observer: DiscreteSolutionObserver):
        self._observers.append(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.update()

    def add_solution(self, time_step: float, solution: np.ndarray):
        super().add_solution(time_step, solution)
        self.notify_observers()
