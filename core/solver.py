from typing import Generic, Optional, Type, TypeVar

import numpy as np
from tqdm.auto import tqdm

from . import time_stepping as ts
from .discrete_solution import DiscreteSolution
from .ode_solver import ExplicitRungeKuttaMethod
from .system import RightHandSide

T = TypeVar("T", bound=DiscreteSolution)


class Solver(Generic[T]):
    name: str
    short: str
    _solution: T
    _time_stepping: ts.TimeStepping
    _ode_solver: ExplicitRungeKuttaMethod[np.ndarray]
    _right_hand_side: RightHandSide
    _cfl_checker: Optional[ts.CFLChecker]

    def __init__(
        self,
        solution: T,
        right_hand_side: RightHandSide,
        ode_solver_type: Type[ExplicitRungeKuttaMethod[np.ndarray]],
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
            self._solution.value,
            start_time=self._solution.time,
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

        except ts.TimeStepTooSmallError:
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
            except ts.CFLConditionViolatedError:
                tqdm.write(
                    f"WARNING: CFL condition is violated at time {self._time_stepping.time:.4f}"
                )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(right_side={self._right_hand_side}, ode_solver={self._ode_solver}, time_stepping={self._time_stepping})"
        )
