import core
import lib
import shallow_water as swe

from .solver import ShallowWaterSolver


def get_lax_friedrichs_flux(
    benchmark: swe.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> lib.LaxFriedrichsFlux:
    swe.assert_constant_bathymetry(benchmark, len(mesh))

    riemann_solver = swe.RiemannSolver()
    return lib.LaxFriedrichsFlux(riemann_solver)


class LaxFriedrichsSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_lax_friedrichs_flux
        super().__init__(benchmark, **kwargs)
