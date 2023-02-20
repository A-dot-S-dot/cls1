"""The fluxes implemented here, are discussed in 'Weel-balanced and energy
stable schemes for the shallow water equations with discontinuous topography' by
U. S. Fjordholm et al."""
import lib
import core
import numpy as np
import shallow_water

from .solver import ShallowWaterSolver


class EnergyStableFlux(lib.NumericalFlux):
    input_dimension = 2
    _gravitational_acceleration: float

    def __init__(self, gravitational_acceleration: float):
        self._gravitational_acceleration = gravitational_acceleration

    def __call__(self, value_left: np.ndarray, value_right: np.ndarray):
        height_average = np.average(
            shallow_water.get_heights(value_left, value_right), axis=0
        )
        velocity_average = np.average(
            shallow_water.get_velocities(value_left, value_right), axis=0
        )

        flux = np.array(
            [
                height_average * velocity_average,
                height_average * velocity_average**2
                + self._gravitational_acceleration / 2 * height_average**2,
            ]
        ).T

        return -flux, flux


def get_energy_stable_flux(
    benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
) -> EnergyStableFlux:
    shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

    return EnergyStableFlux(benchmark.gravitational_acceleration)


class EnergyStableSolver(ShallowWaterSolver):
    def __init__(self, benchmark: shallow_water.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_energy_stable_flux
        super().__init__(benchmark, **kwargs)
