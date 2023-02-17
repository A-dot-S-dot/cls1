import core
import lib
import shallow_water

from ..benchmark import ShallowWaterBenchmark
from .solver import ShallowWaterSolver


class CentralFluxBuilder(lib.NumericalFluxBuilder):
    @staticmethod
    def build_flux(
        benchmark: shallow_water.ShallowWaterBenchmark,
        mesh: core.Mesh,
    ) -> lib.CentralFlux:
        return lib.CentralFlux(shallow_water.Flux(benchmark.gravitational_acceleration))


class CentralFluxSolver(ShallowWaterSolver):
    def _build_flux(
        self, benchmark: ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

        return CentralFluxBuilder.build_flux(benchmark, mesh)
