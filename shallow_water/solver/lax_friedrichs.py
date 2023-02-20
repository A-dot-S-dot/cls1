import core
import numpy as np
import shallow_water as swe

from .solver import ShallowWaterNumericalFlux, ShallowWaterSolver


class LaxFriedrichsFlux(ShallowWaterNumericalFlux):
    input_dimension = 2
    riemann_solver: swe.RiemannSolver

    def __init__(self, gravitational_acceleration: float, bathymetry=None):
        super().__init__(gravitational_acceleration, bathymetry)
        self._riemann_solver = swe.RiemannSolver(gravitational_acceleration)

    def _get_flux(self, value_left, value_right) -> np.ndarray:
        flux, _ = self._riemann_solver.solve(value_left, value_right)

        return flux


def get_lax_friedrichs_flux(
    benchmark: swe.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> LaxFriedrichsFlux:
    bathymetry = swe.build_bathymetry_discretization(benchmark, len(mesh))

    return LaxFriedrichsFlux(
        benchmark.gravitational_acceleration, bathymetry=bathymetry
    )


class LaxFriedrichsSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_lax_friedrichs_flux
        super().__init__(benchmark, **kwargs)
