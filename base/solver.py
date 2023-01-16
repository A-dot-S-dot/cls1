from abc import ABC, abstractmethod
from typing import Optional, Type

from tqdm.auto import tqdm

from base.discretization import DiscreteSolution
from base.ode_solver import ExplicitRungeKuttaMethod
from base.system import SystemVector
from base import time_stepping as ts


class Solver(ABC):
    name: str
    short: str
    _solution: DiscreteSolution
    _time_stepping: ts.TimeStepping
    _ode_solver: ExplicitRungeKuttaMethod
    _right_hand_side: SystemVector
    _cfl_checker: Optional[ts.CFLChecker]

    @abstractmethod
    def __init__(
        self,
        solution: DiscreteSolution,
        right_hand_side: SystemVector,
        ode_solver_type: Type[ExplicitRungeKuttaMethod],
        time_stepping: ts.TimeStepping,
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
            self._solution.initial_data,
            start_time=self._solution.start_time,
        )

    @property
    def solution(self) -> DiscreteSolution:
        return self._solution

    def solve(self, leave_progress_bar=True):
        try:
            for time_step in tqdm(
                self._time_stepping,
                desc=self.name,
                postfix={"DOFs": self._solution.dimension},
                leave=leave_progress_bar,
            ):
                self.update(time_step)

        except ts.TimeStepTooSmallError:
            tqdm.write("WARNING: time step is too small calculation is interrupted")

    def update(self, time_step: float):
        self._ode_solver.execute(time_step)
        self._check_cfl_condition(time_step)
        self._solution.add_solution(time_step, self._ode_solver.solution)

    def _check_cfl_condition(self, time_step: float):
        if self._cfl_checker:
            try:
                self._cfl_checker(time_step, *self._ode_solver.stage_values)
            except ts.CFLConditionViolatedError:
                tqdm.write(
                    f"WARNING: CFL condition is violated at time {self._time_stepping.time:.4f}"
                )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(right_side={self._right_hand_side}, ode_solver={self._ode_solver}, time_stepping={self._time_stepping})"
        )
