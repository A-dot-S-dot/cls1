from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from ode_solver.explicit_runge_kutta import ExplicitRungeKuttaMethod
from system.vector import SystemVector
from system.vector.dof_vector import DOFVector
from tqdm import tqdm

from .time_stepping import TimeStepping


class PDESolver(ABC):
    discrete_solution_dofs: DOFVector
    time_stepping: TimeStepping
    tqdm_kwargs: Dict

    @property
    @abstractmethod
    def solution(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def time(self) -> float:
        ...

    @property
    @abstractmethod
    def time_steps(self) -> int:
        ...

    @abstractmethod
    def update(self, delta_t: float):
        ...

    def solve(self):
        progress_iterator = tqdm(self.time_stepping, **self.tqdm_kwargs)

        for delta_t in progress_iterator:
            self.update(delta_t)


class FiniteElementSolver(PDESolver):
    right_hand_side: SystemVector

    _time: float
    _time_steps: int
    _time_stepping: TimeStepping
    _ode_solver: ExplicitRungeKuttaMethod

    @property
    def solution(self) -> np.ndarray:
        return self._ode_solver.solution

    @property
    def time(self) -> float:
        return self._time

    @property
    def time_steps(self) -> int:
        return self._time_steps

    @property
    def time_stepping(self) -> TimeStepping:
        return self._time_stepping

    @time_stepping.setter
    def time_stepping(self, time_stepping):
        self._time_stepping = time_stepping
        self._time = time_stepping.start_time
        self._time_steps = 0

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

    def _ode_right_hand_side_function(self, dofs: np.ndarray) -> np.ndarray:
        if (dofs != self.ode_solver.solution).any():
            self.discrete_solution_dofs.dofs = dofs

            if not self.time_stepping.satisfy_cfl():
                tqdm.write(
                    f"WARNING: CFL condition is violated at time {self._time:.4f}"
                )

        if not self.time_stepping.satisfy_cfl():
            tqdm.write(f"WARNING: CFL condition is violated at time {self._time:.4f}")

        return self.right_hand_side.values.copy()

    def update(self, delta_t: float):
        self.ode_solver.execute_step(delta_t)
        self._time += delta_t
        self._time_steps += 1

        # update all DOF dependent quantities using observer pattern
        self.discrete_solution_dofs.dofs = self.ode_solver.solution
