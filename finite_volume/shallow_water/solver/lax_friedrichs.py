import finite_volume
import finite_volume.shallow_water as swe


class LaxFriedrichsFluxGetter(swe.FluxGetter):
    def _get_flux(
        self, benchmark: swe.ShallowWaterBenchmark
    ) -> finite_volume.NumericalFlux:
        riemann_solver = swe.RiemannSolver(benchmark.gravitational_acceleration)
        return finite_volume.LaxFriedrichsFlux(riemann_solver)


class LaxFriedrichsSolver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self.flux_getter = LaxFriedrichsFluxGetter()
        super().__init__(benchmark, **kwargs)


class LaxFriedrichsParser(finite_volume.SolverParser):
    prog = "llf"
    name = "Lax-Friedrichs finite volume scheme"
    solver = LaxFriedrichsSolver

    def _add_arguments(self):
        self._add_ode_solver()
