"""The fluxes implemented here, are discussed in 'Weel-balanced and energy
stable schemes for the shallow water equations with discontinuous topography' by
U. S. Fjordholm et al."""
from typing import Tuple

import core
import numpy as np
import shallow_water as swe

from .solver import ShallowWaterNumericalFlux, ShallowWaterSolver


class EnergyStableFlux(ShallowWaterNumericalFlux):
    input_dimension = 2

    def __init__(
        self, gravitational_acceleration: float, step_length: float, bathymetry=None
    ):
        super().__init__(gravitational_acceleration, bathymetry)
        self._step_length = step_length
        self._flux = swe.Flux(gravitational_acceleration)

    def _get_flux(self, value_left: np.ndarray, value_right: np.ndarray) -> np.ndarray:
        average = swe.get_average(value_left, value_right)
        return self._flux(average)


def get_energy_stable_flux(
    benchmark: swe.ShallowWaterBenchmark, mesh: core.Mesh
) -> EnergyStableFlux:
    bathymetry = swe.build_bathymetry_discretization(benchmark, len(mesh))

    return EnergyStableFlux(
        benchmark.gravitational_acceleration, mesh.step_length, bathymetry=bathymetry
    )


class EnergyStableSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_energy_stable_flux
        super().__init__(benchmark, **kwargs)
