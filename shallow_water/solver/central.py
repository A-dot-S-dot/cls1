import core
import lib
import shallow_water

from ..benchmark import ShallowWaterBenchmark
from .solver import ShallowWaterSolver


def get_central_flux(
    benchmark: shallow_water.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> lib.CentralFlux:
    return lib.CentralFlux(shallow_water.Flux(benchmark.gravitational_acceleration))


class CentralFluxSolver(ShallowWaterSolver):
    def _get_flux(
        self, benchmark: ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

        return get_central_flux(benchmark, mesh)
