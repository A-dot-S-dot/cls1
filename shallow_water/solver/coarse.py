from typing import Dict

import core
import shallow_water as swe
import defaults

from .lax_friedrichs import LaxFriedrichsFluxGetter
from .solver import ShallowWaterSolver


class CoarseSolver(ShallowWaterSolver):
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
