from abc import ABC, abstractmethod
from typing import Dict
from tqdm import tqdm

import numpy as np
from ode_solver.explicit_runge_kutta import ExplicitRungeKuttaMethod
from system.vector import DOFVector
from .time_stepping import TimeStepping


class PDESolver(ABC):
    discrete_solution_dofs: DOFVector
    tqdm_kwargs: Dict

    _time: float
    _time_stepping: TimeStepping
    _ode_solver: ExplicitRungeKuttaMethod

    @property
    def solution(self) -> np.ndarray:
        return self._ode_solver.solution

    @property
    def time(self) -> float:
        return self._time

    @property
    def time_stepping(self) -> TimeStepping:
        return self._time_stepping

    @time_stepping.setter
    def time_stepping(self, time_stepping):
        self._time_stepping = time_stepping
        self._time = time_stepping.start_time

    @property
    def ode_solver(self) -> ExplicitRungeKuttaMethod:
        return self._ode_solver

    @ode_solver.setter
    def ode_solver(self, ode_solver: ExplicitRungeKuttaMethod):
        self._ode_solver = ode_solver
        self._ode_solver.time = self.time_stepping.start_time
        self._ode_solver.set_start_value(self.discrete_solution_dofs.dofs.copy())
        self._ode_solver.right_hand_side_function = (
            lambda dofs: self._ode_right_hand_side_function(dofs)
        )

    @abstractmethod
    def _ode_right_hand_side_function(self, dofs: np.ndarray) -> np.ndarray:
        ...

    def update(self, delta_t: float):
        self._time += delta_t
        self.ode_solver.execute_step(delta_t)

    def solve(self):
        progress_iterator = tqdm(self.time_stepping, **self.tqdm_kwargs)

        for delta_t in progress_iterator:
            self.update(delta_t)
