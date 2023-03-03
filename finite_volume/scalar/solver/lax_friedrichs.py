import core
import finite_volume
from finite_volume import scalar


class LaxFriedrichsFluxGetter(finite_volume.FluxGetter):
    def __call__(
        self, benchmark: core.Benchmark, space: finite_volume.FiniteVolumeSpace
    ) -> finite_volume.NumericalFlux:
        riemann_solver = scalar.get_riemann_solver(benchmark.problem)
        return finite_volume.LaxFriedrichsFlux(riemann_solver)


class LaxFriedrichsSolver(finite_volume.Solver):
    def __init__(self, benchmark: core.Benchmark, **kwargs):
        self.flux_getter = LaxFriedrichsFluxGetter()
        super().__init__(benchmark, **kwargs)
