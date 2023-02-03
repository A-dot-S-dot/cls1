from abc import ABC, abstractmethod
from typing import Any


class SourceTermDiscretization(ABC):
    @property
    @abstractmethod
    def step_length(self) -> float:
        ...

    @abstractmethod
    def __call__(self, height_left, height_right, topography_step) -> Any:
        ...


class VanishingSourceTerm(SourceTermDiscretization):
    @property
    def step_length(self) -> float:
        return 0.0

    def __call__(self, height_left, height_right, topography_step) -> Any:
        return 0.0


class NaturalSouceTerm(SourceTermDiscretization):
    """Discretization of h*Db, i.e.

    h*Db=(hL+hR)/(2*step_length)*topography_step.

    """

    _step_length: float

    def __init__(self, step_length: float):
        self._step_length = step_length

    @property
    def step_length(self) -> float:
        return self._step_length

    def __call__(self, height_left, height_right, topography_step) -> Any:
        return (height_left + height_right) / (2 * self._step_length) * topography_step


def build_source_term(step_length: float) -> SourceTermDiscretization:
    return NaturalSouceTerm(step_length)
