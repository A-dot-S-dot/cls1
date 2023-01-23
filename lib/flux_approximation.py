from typing import Callable

import numpy as np
from core import SystemVector


class FluxApproximation:
    """Group Finite Element Approximation (GFE) of f(v) with finite elements,
    where f is a flux and v a finite element. To be more precise it is defined
    as following:

        F = sum(fi*bi)

    where fi = f(vi) and bi denotes the basis elements of the finite element
    space.

    Not assembled by default.

    """

    _flux: Callable[[np.ndarray], np.ndarray]
    _flux_approximation: np.ndarray
    _last_dof_vector = np.empty(1)

    def __init__(self, flux: Callable[[np.ndarray], np.ndarray]):
        self._flux = flux

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        if (dof_vector != self._last_dof_vector).any():
            self._flux_approximation = self._flux(dof_vector)
            self._last_dof_vector = dof_vector

        return self._flux_approximation


def build_flux_approximation(problem: str) -> SystemVector:
    fluxes = {"advection": lambda u: u, "burgers": lambda u: 1 / 2 * u**2}
    return FluxApproximation(fluxes[problem])
