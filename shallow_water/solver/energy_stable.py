"""The fluxes implemented here, are discussed in 'Weel-balanced and energy
stable schemes for the shallow water equations with discontinuous topography' by
U. S. Fjordholm et al."""
from typing import Tuple

import lib
import numpy as np
import shallow_water as swe

from .solver import ShallowWaterSolver, FluxGetter


class EnergyStableFlux(lib.NumericalFlux):
    input_dimension = 2
    _flux: swe.Flux

    def __init__(self, gravitational_acceleration=None):
        self._flux = swe.Flux(gravitational_acceleration)

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        average = swe.get_average(value_left, value_right)
        flux = self._flux(average)
        return -flux, flux


class EnergyStableFluxGetter(FluxGetter):
    def _get_flux(self, benchmark: swe.ShallowWaterBenchmark) -> lib.NumericalFlux:
        return EnergyStableFlux(benchmark.gravitational_acceleration)


class EnergyStableSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = EnergyStableFluxGetter()
        super().__init__(benchmark, **kwargs)
