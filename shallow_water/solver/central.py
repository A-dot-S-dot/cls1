import core
import lib
import shallow_water

from .solver import ShallowWaterSolver


def get_central_flux(
    benchmark: shallow_water.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> lib.CentralFlux:
    shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

    return lib.CentralFlux(shallow_water.Flux(benchmark.gravitational_acceleration))


class CentralFluxSolver(ShallowWaterSolver):
    def __init__(self, benchmark: shallow_water.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_central_flux
        super().__init__(benchmark, **kwargs)
