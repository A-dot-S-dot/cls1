from typing import Dict

import core
import shallow_water
import defaults

from .lax_friedrichs import get_lax_friedrichs_flux
from .solver import ShallowWaterSolver


class CoarseSolver(ShallowWaterSolver):
    _coarsening_degree: int

    def _build_args(
        self,
        benchmark: shallow_water.ShallowWaterBenchmark,
        coarsening_degree=None,
        flux_getter=None,
        **kwargs
    ) -> Dict:
        self._get_flux = flux_getter or get_lax_friedrichs_flux
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
