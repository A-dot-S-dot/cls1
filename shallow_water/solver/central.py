import core
import lib
import shallow_water as swe

from .solver import ShallowWaterSolver


def get_central_flux(
    benchmark: swe.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> lib.CentralFlux:
    swe.assert_constant_bathymetry(benchmark, len(mesh))

    return lib.CentralFlux(swe.Flux(benchmark.gravitational_acceleration))


class CentralFluxSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_central_flux
        super().__init__(benchmark, **kwargs)
