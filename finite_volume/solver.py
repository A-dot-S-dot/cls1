from abc import ABC, abstractmethod
from typing import Dict, Optional

import core
import defaults

from .space import FiniteVolumeSpace, get_finite_volume_solution
from .numerical_flux import NumericalFlux, NumericalFluxDependentRightHandSide


class FluxGetter(ABC):
    @abstractmethod
    def __call__(
        self, benchmark: core.Benchmark, space: FiniteVolumeSpace
    ) -> NumericalFlux:
        ...


class Solver(core.Solver):
    flux_getter: FluxGetter
    _get_boundary_conditions = core.get_boundary_conditions

    def __init__(self, benchmark: core.Benchmark, **solver_args):
        solver_args = self._get_solver_args(benchmark, **solver_args)
        core.Solver.__init__(self, **solver_args)

    def _get_solver_args(
        self,
        benchmark: core.Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        ode_solver_type=None,
        save_history=False,
    ) -> Dict:
        solution = get_finite_volume_solution(benchmark, mesh_size, save_history)
        step_length = solution.space.mesh.step_length

        numerical_flux = self.flux_getter(benchmark, solution.space)
        boundary_conditions = self._get_boundary_conditions(
            *benchmark.boundary_conditions,
            radius=numerical_flux.input_dimension // 2,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        right_hand_side = NumericalFluxDependentRightHandSide(
            numerical_flux, step_length, boundary_conditions
        )

        time_stepping = core.get_mesh_dependent_time_stepping(
            benchmark,
            solution.space.mesh,
            cfl_number or defaults.FINITE_VOLUME_CFL_NUMBER,
        )

        return {
            "solution": solution,
            "right_hand_side": right_hand_side,
            "ode_solver_type": ode_solver_type or core.Heun,
            "time_stepping": time_stepping,
            "name": name,
            "short": short,
        }

    def reinitialize(self, benchmark: core.Benchmark):
        initial_data = get_finite_volume_solution(
            benchmark, len(self.solution.space.mesh)
        )
        self.solution.set_value(initial_data.value, initial_data.time)
        self._ode_solver.reinitialize(initial_data.value, initial_data.time)


class SolverParser(core.SolverParser):
    _flux_getter: Optional[Dict]
    _cfl_default = defaults.FINITE_VOLUME_CFL_NUMBER

    def __init__(self, flux_getter=None):
        self._flux_getter = flux_getter
        super().__init__()

    def _add_flux(self):
        assert self._flux_getter is not None

        self.add_argument(
            "+f",
            "++flux",
            help="""Choose flux by key. Available fluxes are: """
            + ", ".join([*self._flux_getter.keys()]),
            type=lambda input: self._flux_getter[input],
            metavar="<flux>",
            dest="flux_getter",
        )
