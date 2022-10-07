from abc import ABC, abstractmethod

import numpy as np
from custom_type import ScalarFunction


class Interpolator(ABC):
    """Interpolates scalar functions by returning DOF Vectors."""

    @abstractmethod
    def interpolate(self, f: ScalarFunction) -> np.ndarray:
        ...
