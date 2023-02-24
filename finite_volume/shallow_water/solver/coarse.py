from typing import Dict

import core
import defaults
import finite_volume.shallow_water as swe

from .lax_friedrichs import LaxFriedrichsFluxGetter


class CoarseSolver(swe.Solver):
    _coarsening_degree: int

    def _build_args(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        coarsening_degree=None,
        flux_getter=None,
        **kwargs
    ) -> Dict:
        self._get_flux = flux_getter or LaxFriedrichsFluxGetter()
        self._coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE

        return super()._build_args(benchmark, **kwargs)

    @property
    def solution(self) -> core.DiscreteSolution:
        if isinstance(self._solution, core.DiscreteSolutionWithHistory):
            return core.CoarseSolutionWithHistory(
                self._solution, self._coarsening_degree
            )
        else:
            return core.CoarseSolution(self._solution, self._coarsening_degree)
