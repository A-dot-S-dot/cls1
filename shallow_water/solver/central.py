from typing import Callable

import core
import numpy as np
import shallow_water as swe

from .solver import ShallowWaterNumericalFlux, ShallowWaterSolver


class CentralFlux(ShallowWaterNumericalFlux):
    input_dimension = 2
    _flux: Callable

    def __init__(self, gravitational_acceleration: float, bathymetry=None):
        super().__init__(gravitational_acceleration, bathymetry)
        self._flux = swe.Flux(gravitational_acceleration)

    def _get_flux(self, value_left: np.ndarray, value_right: np.ndarray) -> np.ndarray:
        return (self._flux(value_left) + self._flux(value_right)) / 2


def get_central_flux(
    benchmark: swe.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> CentralFlux:
    bathymetry = swe.build_bathymetry_discretization(benchmark, len(mesh))

    return CentralFlux(benchmark.gravitational_acceleration, bathymetry=bathymetry)


class CentralFluxSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_central_flux
        super().__init__(benchmark, **kwargs)
