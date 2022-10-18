from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .discrete_solution import DiscreteSolution


class DiscreteSolutionObserver(ABC):
    _discrete_solution: DiscreteSolution

    def __init__(self, observable: "DiscreteSolutionObservable"):
        observable.register_observer(self)
        self._discrete_solution = observable

    @abstractmethod
    def update(self):
        ...


class DiscreteSolutionObservable(DiscreteSolution):
    """Every time discrete solution is changed it informes observers."""

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
