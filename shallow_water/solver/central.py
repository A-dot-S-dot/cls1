import lib
import shallow_water as swe

from .solver import FluxGetter, ShallowWaterSolver


class CentralFluxGetter(FluxGetter):
    def _get_flux(self, benchmark: swe.ShallowWaterBenchmark) -> lib.NumericalFlux:
        flux = swe.Flux(benchmark.gravitational_acceleration)
        return lib.CentralFlux(flux)


class CentralFluxSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = CentralFluxGetter()
        super().__init__(benchmark, **kwargs)
