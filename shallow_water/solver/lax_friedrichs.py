import core
import lib
import shallow_water

from .solver import ShallowWaterSolver


def get_lax_friedrichs_flux(
    benchmark: shallow_water.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> lib.LaxFriedrichsFlux:
    shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

    riemann_solver = shallow_water.RiemannSolver()
    return lib.LaxFriedrichsFlux(riemann_solver)


class LaxFriedrichsSolver(ShallowWaterSolver):
    def __init__(self, benchmark: shallow_water.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_lax_friedrichs_flux
        super().__init__(benchmark, **kwargs)
