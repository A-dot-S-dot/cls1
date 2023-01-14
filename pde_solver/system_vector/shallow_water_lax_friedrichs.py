from typing import Callable, Tuple

import numpy as np

from .numerical_flux import NumericalFlux

ScalarFunction = Callable[[float], float]


class ShallowWaterLaxFriedrichsFlux(NumericalFlux):
    """WARNING: Only designed for flat bottoms."""

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super().__call__(dof_vector)
