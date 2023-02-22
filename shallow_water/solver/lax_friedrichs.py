import lib
import shallow_water as swe

from .solver import ShallowWaterSolver, FluxGetter


class LaxFriedrichsFluxGetter(FluxGetter):
    def _get_flux(self, benchmark: swe.ShallowWaterBenchmark) -> lib.NumericalFlux:
        riemann_solver = swe.RiemannSolver(benchmark.gravitational_acceleration)
        return lib.LaxFriedrichsFlux(riemann_solver)


class LaxFriedrichsSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = LaxFriedrichsFluxGetter()
        super().__init__(benchmark, **kwargs)
