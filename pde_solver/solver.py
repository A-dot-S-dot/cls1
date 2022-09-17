from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from ode_solver.explicit_runge_kutta import ExplicitRungeKuttaMethod
from system.vector import DOFVector


class PDESolver(ABC):
    discrete_solution_dofs: DOFVector
    tqdm_kwargs: Dict
    _ode_solver: ExplicitRungeKuttaMethod

    @property
    @abstractmethod
    def time(self) -> float:
        ...

    @abstractmethod
    def solve(self, target_time: float, time_steps_number: int):
        ...

    @property
    def solution(self) -> np.ndarray:
        return self._ode_solver.solution

    @property
    def ode_solver(self) -> ExplicitRungeKuttaMethod:
        return self._ode_solver

    @ode_solver.setter
    def ode_solver(self, ode_solver: ExplicitRungeKuttaMethod):
        self._ode_solver = ode_solver
        self._ode_solver.time = self.time
        self._ode_solver.set_start_value(self.discrete_solution_dofs.dofs.copy())
        self._ode_solver.right_hand_side_function = (
            lambda dofs: self._ode_right_hand_side_function(dofs)
        )

    @abstractmethod
    def _ode_right_hand_side_function(self, dofs: np.ndarray) -> np.ndarray:
        ...
