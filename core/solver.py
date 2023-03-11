import argparse
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Sequence, Type, TypeVar

import defaults
import numpy as np
from tqdm.auto import tqdm

from . import ode_solver, time_stepping
from .benchmark import Benchmark
from .discrete_solution import DiscreteSolution
from .system import RightHandSide
from .type import *

T = TypeVar("T", bound=DiscreteSolution)


class Solver(Generic[T], ABC):
    name: str
    short: str
    _solution: T
    _time_stepping: time_stepping.TimeStepping
    _ode_solver: ode_solver.ExplicitRungeKuttaMethod[np.ndarray]
    _right_hand_side: RightHandSide
    _cfl_checker: Optional[time_stepping.CFLChecker]

    def __init__(
        self,
        solution: T,
        right_hand_side: RightHandSide,
        ode_solver_type: Type[ode_solver.ExplicitRungeKuttaMethod[np.ndarray]],
        time_stepping: time_stepping.TimeStepping,
        name=None,
        short=None,
        cfl_checker=None,
    ):
        self._solution = solution
        self._right_hand_side = right_hand_side
        self._time_stepping = time_stepping
        self.name = name or "Solver"
        self.short = short or ""
        self._cfl_checker = cfl_checker
        self._ode_solver = ode_solver_type(
            self._right_hand_side,
            self._solution.value,
            initial_time=self._solution.time,
        )

    @property
    def solution(self) -> T:
        return self._solution

    def solve(self, **tqdm_kwargs):
        try:
            for time_step in tqdm(
                self._time_stepping,
                desc=self.name,
                postfix={"DOFs": self._solution.dimension},
                **tqdm_kwargs,
            ):
                self.update(time_step)

        except time_stepping.TimeStepTooSmallError:
            tqdm.write("WARNING: time step is too small calculation is interrupted")

    def update(self, time_step: float):
        self._ode_solver.execute(time_step)
        self._check_cfl_condition(time_step)
        self._solution.update(time_step, self._ode_solver.solution)

    def _check_cfl_condition(self, time_step: float):
        if self._cfl_checker:
            try:
                self._cfl_checker(
                    time_step,
                    self._ode_solver.time_nodes,
                    *self._ode_solver.stage_values,
                )
            except time_stepping.CFLConditionViolatedError:
                tqdm.write(
                    f"WARNING: CFL condition is violated at time {self._time_stepping.time:.4f}"
                )

    @abstractmethod
    def reinitialize(self, benchmark: Benchmark):
        ...

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(right_side={self._right_hand_side}, ode_solver={self._ode_solver}, time_stepping={self._time_stepping})"
        )


class SolverParser(argparse.ArgumentParser):
    prog: str
    name: str
    solver: Type[Solver]
    _cfl_default = 0.5
    _mesh_size_default = defaults.CALCULATE_MESH_SIZE

    def __init__(self):
        argparse.ArgumentParser.__init__(
            self,
            prog=self.prog,
            description=self.name,
            prefix_chars="+",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_standard_arguments()
        self._add_arguments()

    def _add_standard_arguments(self):
        self._add_name()
        self._add_short()
        self._add_mesh_size()
        self._add_cfl()

    def _add_arguments(self):
        ...

    def _add_name(self):
        self.add_argument(
            "+n",
            "++name",
            help="Specify short name",
            metavar="<name>",
            default=self.name,
        )

    def _add_short(self):
        self.add_argument(
            "+s",
            "++short",
            help="Specify short name",
            metavar="<short>",
            default=self.prog,
        )

    def _add_mesh_size(self, default=None):
        self.add_argument(
            "+m",
            "++mesh-size",
            help="Number of mesh cells. If not specified use the default size for the chosen task.",
            type=positive_int,
            metavar="<size>",
            default=default or self._mesh_size_default,
        )

    def _add_cfl(self, default=None):
        self.add_argument(
            "++cfl",
            help="Specify the cfl number for time stepping.",
            type=positive_float,
            metavar="<number>",
            dest="cfl_number",
            default=default or self._cfl_default,
        )

    def _add_adaptive_time_stepping(self):
        self.add_argument(
            "++adaptive",
            help="Make time stepping adaptive, if available.",
            action="store_true",
        )

    def _add_ode_solver(self):
        _ode_solver = {
            "euler": ode_solver.ForwardEuler,
            "heun": ode_solver.Heun,
            "ssp3": ode_solver.StrongStabilityPreservingRungeKutta3,
        }

        self.add_argument(
            "+o",
            "++ode-solver",
            help="Specify ode solver. Available solver are: "
            + ", ".join([*_ode_solver.keys()]),
            type=lambda input: _ode_solver[input],
            metavar="<solver>",
            dest="ode_solver_type",
            default=_ode_solver[defaults.ODE_SOLVER],
        )

    def parse_arguments(self, *arguments) -> argparse.Namespace:
        arguments = self.parse_args(*arguments)
        arguments.solver = self.solver

        return arguments


class SolverAction(argparse.Action):
    solver_parsers: Dict

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: List[str],
        option_string: Optional[str] = ...,
    ) -> None:
        solver_arguments = list()
        while len(values) > 0:
            raw_solver_arguments = self._pop_solver_arguments(values)
            solver_arguments.append(self._get_solver_namespace(raw_solver_arguments))

        setattr(namespace, "solver", solver_arguments)

    def _pop_solver_arguments(self, values: List[str]) -> List[str]:
        slice_index = None

        for i, value in enumerate(values):
            if (
                i > 0
                and value in self.solver_parsers.keys()
                and values[i - 1] not in ["+f"]
            ):
                slice_index = i
                break

        solver_arguments = values[:slice_index]
        del values[:slice_index]

        return solver_arguments

    def _get_solver_namespace(
        self, raw_solver_arguments: Sequence[str]
    ) -> argparse.Namespace:
        solver_key = raw_solver_arguments[0]
        namespace = argparse.Namespace()
        try:
            self.solver_parsers[solver_key].parse_arguments(
                raw_solver_arguments[1:], namespace
            )
        except KeyError:
            print(
                "ERROR: Only the following solvers are available: "
                + ", ".join(self.solver_parsers.keys())
            )
            quit()

        return namespace
