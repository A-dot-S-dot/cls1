import numpy as np
from system.matrix import SystemMatrix
from system.vector import SystemVector
from tqdm import trange

from .solver import PDESolver


class ContinuousGalerkinSolver(PDESolver):
    mass: SystemMatrix
    flux_gradient: SystemVector
    time: float = 0

    def _ode_right_hand_side_function(self, dofs: np.ndarray) -> np.ndarray:
        # update all DOF dependent quantities using observer pattern
        self.discrete_solution_dofs.dofs = dofs

        return self.mass.inverse(self.flux_gradient.values)

    def update(self, delta_t: float):
        self.time += delta_t
        self.ode_solver.execute_step(delta_t)

    def solve(self, target_time: float, time_steps_number: int):
        time_grid = np.linspace(self.time, target_time, time_steps_number + 1)
        progress_iterator = trange(len(time_grid) - 1, **self.tqdm_kwargs)

        for i in progress_iterator:
            delta_t = time_grid[i + 1] - time_grid[i]
            self.update(delta_t)
