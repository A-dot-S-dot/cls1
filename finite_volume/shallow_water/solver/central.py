import finite_volume
import finite_volume.shallow_water as swe


class CentralFluxGetter(swe.FluxGetter):
    def _get_flux(
        self, benchmark: swe.ShallowWaterBenchmark
    ) -> finite_volume.NumericalFlux:
        flux = swe.Flux(benchmark.gravitational_acceleration)
        return finite_volume.CentralFlux(flux)


class CentralFluxSolver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self.flux_getter = CentralFluxGetter()
        super().__init__(benchmark, **kwargs)


class CentralFluxParser(finite_volume.SolverParser):
    prog = "central"
    name = "Central scheme"
    solver = CentralFluxSolver
