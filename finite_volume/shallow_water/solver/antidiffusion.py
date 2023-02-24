from typing import Dict

import core
import defaults
import finite_volume
import finite_volume.shallow_water as swe

from .lax_friedrichs import LaxFriedrichsFluxGetter


class LinearAntidiffusiveSolver(swe.Solver):
    _gamma: float
    _get_raw_flux: swe.FluxGetter

    def _build_args(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        gamma=None,
        flux_getter=None,
        **kwargs
    ) -> Dict:
        self._gamma = gamma or defaults.ANTIDIFFUSION_GAMMA
        self._get_raw_flux = flux_getter or LaxFriedrichsFluxGetter()

        return super()._build_args(benchmark, **kwargs)

    def _get_flux(
        self, benchmark: swe.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> finite_volume.NumericalFlux:
        numerical_flux = self._get_raw_flux(benchmark, mesh)
        antidiffusive_flux = finite_volume.LinearAntidiffusiveFlux(
            self._gamma, mesh.step_length
        )

        return finite_volume.CorrectedNumericalFlux(numerical_flux, antidiffusive_flux)
