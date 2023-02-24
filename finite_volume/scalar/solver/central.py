import core
import finite_volume
from finite_volume import scalar


class CentralFluxGetter(finite_volume.FluxGetter):
    def __call__(
        self, benchmark: core.Benchmark, space: finite_volume.FiniteVolumeSpace
    ) -> finite_volume.NumericalFlux:
        return finite_volume.CentralFlux(scalar.get_flux(benchmark.problem))


class CentralFluxSolver(finite_volume.Solver):
    def __init__(self, benchmark: core.Benchmark, **kwargs):
        self._get_flux = CentralFluxGetter()
        super().__init__(benchmark, **kwargs)
