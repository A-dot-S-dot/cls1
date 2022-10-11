from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .discrete_solution import DiscreteSolution


class DiscreteSolutionObserver(ABC):
    @abstractmethod
    def update(self):
        ...


class DiscreteSolutionObservable(DiscreteSolution):
    """Every time discrete solution is changed it informes observers."""

    _observers: List[DiscreteSolutionObserver]

    def __init__(self, discrete_solution: DiscreteSolution):
        self.solution = discrete_solution.solution
        self.time = discrete_solution.time

        self._observers = []

    def register_observer(self, observer: DiscreteSolutionObserver):
        self._observers.append(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.update()

    def add_solution(self, time_step: float, solution: np.ndarray):
        super().add_solution(time_step, solution)
        self.notify_observers()
