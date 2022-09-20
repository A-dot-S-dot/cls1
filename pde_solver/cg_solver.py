import numpy as np
from system.matrix import SystemMatrix
from system.vector import SystemVector

from .solver import PDESolver


class ContinuousGalerkinSolver(PDESolver):
    mass: SystemMatrix
    flux_gradient: SystemVector

    def _ode_right_hand_side_function(self, dofs: np.ndarray) -> np.ndarray:
        # update all DOF dependent quantities using observer pattern
        self.discrete_solution_dofs.dofs = dofs

        return self.mass.inverse(self.flux_gradient.values)
