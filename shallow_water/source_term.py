from abc import ABC, abstractmethod
from typing import Any


class SourceTermDiscretization(ABC):
    @abstractmethod
    def __call__(self, height_left, height_right, topography_step, step_length) -> Any:
        ...


class VanishingSourceTerm(SourceTermDiscretization):
    def __call__(self, height_left, height_right, topography_step, step_length) -> Any:
        return 0


class NaturalSouceTerm(SourceTermDiscretization):
    """Discretization of h*Db, i.e.

    h*Db=(hL+hR)/(2*step_length)*topography_step.

    """

    def __call__(self, height_left, height_right, topography_step, step_length) -> Any:
        return (height_left + height_right) / (2 * step_length) * topography_step


def build_source_term() -> SourceTermDiscretization:
    return NaturalSouceTerm()
