from typing import Any, List

import numpy as np
from fem import FiniteElementSpace

from .system_vector import SystemVector


class DOFVector(SystemVector):
    _element_space: FiniteElementSpace
    _observers: List[Any]

    def __init__(self, element_space: FiniteElementSpace):
        super().__init__(element_space)
        self._observers = []

    @property
    def dofs(self) -> np.ndarray:
        return self.values

    @dofs.setter
    def dofs(self, dofs: np.ndarray):
        self._values = dofs
        self.update_observers()

    def register_observer(self, observer):
        self._observers.append(observer)

    def update_observers(self):
        for observer in self._observers:
            observer.update()
