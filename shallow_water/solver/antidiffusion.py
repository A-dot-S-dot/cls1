from typing import Dict

import core
import defaults
import lib
import shallow_water

from .solver import ShallowWaterSolver
from .lax_friedrichs import get_lax_friedrichs_flux


class LinearAntidiffusiveSolver(ShallowWaterSolver):
    _gamma: float
    _get_raw_flux: lib.FLUX_GETTER[shallow_water.ShallowWaterBenchmark]

    def _build_args(
        self,
        benchmark: shallow_water.ShallowWaterBenchmark,
        gamma=None,
        flux_getter=None,
        **kwargs
    ) -> Dict:
        self._gamma = gamma or defaults.ANTIDIFFUSION_GAMMA
        self._get_raw_flux = flux_getter or get_lax_friedrichs_flux

        return super()._build_args(benchmark, **kwargs)

    def _get_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        numerical_flux = self._get_raw_flux(benchmark, mesh)
        antidiffusive_flux = lib.LinearAntidiffusiveFlux(self._gamma, mesh.step_length)

        return lib.CorrectedNumericalFlux(numerical_flux, antidiffusive_flux)
