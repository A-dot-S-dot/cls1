import numpy as np
from pde_solver.system_matrix import SystemMatrix

from .system_vector import SystemVector


class CGRightHandSide(SystemVector):
    """Right hand side of continuous Galerkin method r. To be more
    precise it is defined as following:

       Mr = A

    where M denotes mass marix and A the discrete flux gradient.

    """

    mass: SystemMatrix
    flux_gradient: SystemVector

    def __init__(self, mass: SystemMatrix, flux_gradient: SystemVector):
        self.mass = mass
        self.flux_gradient = flux_gradient

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return self.mass.inverse(self.flux_gradient(dof_vector))
