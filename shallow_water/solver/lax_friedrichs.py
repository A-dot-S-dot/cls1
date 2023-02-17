from typing import Dict

import core
import defaults
import lib
import shallow_water

from ..benchmark import ShallowWaterBenchmark
from .solver import ShallowWaterSolver


class LaxFriedrichsFluxBuilder(lib.NumericalFluxBuilder):
    @staticmethod
    def build_flux(
        benchmark: shallow_water.ShallowWaterBenchmark,
        mesh: core.Mesh,
    ) -> lib.LaxFriedrichsFlux:
        riemann_solver = shallow_water.RiemannSolver()
        return lib.LaxFriedrichsFlux(riemann_solver)


class LaxFriedrichsSolver(ShallowWaterSolver):
    def _build_flux(
        self, benchmark: ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

        return LaxFriedrichsFluxBuilder.build_flux(benchmark, mesh)


class CoarseLocalLaxFriedrichsSolver(LaxFriedrichsSolver, core.CoarseSolver):
    def __init__(self, *args, coarsening_degree=None, **kwargs):
        self._coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        LaxFriedrichsSolver.__init__(self, *args, **kwargs)


class AntidiffusiveLocalLaxFriedrichsSolver(ShallowWaterSolver):
    def _build_args(
        self, benchmark: shallow_water.ShallowWaterBenchmark, gamma=None, **kwargs
    ) -> Dict:
        self._gamma = gamma or defaults.ANTIDIFFUSION_GAMMA

        return super()._build_args(benchmark, **kwargs)

    def _build_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

        numerical_flux = LaxFriedrichsFluxBuilder.build_flux(benchmark, mesh)
        antidiffusive_flux = lib.LinearAntidiffusiveFlux(self._gamma, mesh.step_length)

        return lib.CorrectedNumericalFlux(numerical_flux, antidiffusive_flux)
