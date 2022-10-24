from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from tqdm import tqdm

from pde_solver.discrete_solution import (
    DiscreteSolutionObservable,
    CoarseSolution,
    DiscreteSolution,
)
from pde_solver.ode_solver import ExplicitRungeKuttaMethod
from pde_solver.solver_space import FiniteVolumeSpace, SolverSpace
from pde_solver.system_flux import SystemFlux
from pde_solver.system_vector import SystemVector
from pde_solver.time_stepping import TimeStepping, TimeStepTooSmallError


class PDESolver(ABC):
    solver_space: SolverSpace
    solution: DiscreteSolutionObservable
    time_stepping: TimeStepping
    tqdm_kwargs: Dict

    def solve(self):
        progress_iterator = tqdm(self.time_stepping, **self.tqdm_kwargs)

        for _ in progress_iterator:
            try:
                self.update()
            except TimeStepTooSmallError:
                tqdm.write("WARNING: time step is too small calculation is interrupted")
                break

    @abstractmethod
    def update(self):
        ...


class ScalarFiniteElementSolver(PDESolver):
    right_hand_side: SystemVector
    ode_solver: ExplicitRungeKuttaMethod

    def update(self):
        time_step = self.time_stepping.time_step

        self.ode_solver.execute(time_step)
        self.solution.add_solution(time_step, self.ode_solver.solution)


class FiniteVolumeSolver(PDESolver):
    solver_space: FiniteVolumeSpace
    numerical_flux: SystemFlux
    time_step: float
    left_flux: np.ndarray
    right_flux: np.ndarray

    def update(self):
        self.time_step = self.time_stepping.time_step
        self.left_flux, self.right_flux = self.numerical_flux(self.solution.end_values)

        step_length = self.solver_space.mesh.step_length
        updated_solution = (
            self.solution.end_values
            + self.time_step * (self.right_flux + -self.left_flux) / step_length
        )

        self.solution.add_solution(self.time_step, updated_solution)


class ExactCoarseSolver(FiniteVolumeSolver):
    coarsening_degree: int

    _fine_solution: DiscreteSolutionObservable

    @property
    def solution(self) -> DiscreteSolution:
        return CoarseSolution(self._fine_solution, self.coarsening_degree)

    @solution.setter
    def solution(self, solution: DiscreteSolutionObservable):
        self._fine_solution = solution

    def update(self):
        self.time_step = self.time_stepping.time_step
        self.left_flux, self.right_flux = self.numerical_flux(
            self._fine_solution.end_values
        )

        step_length = self.solver_space.mesh.step_length
        updated_solution = (
            self._fine_solution.end_values
            + self.time_step * (self.right_flux + -self.left_flux) / step_length
        )

        self._fine_solution.add_solution(self.time_step, updated_solution)
