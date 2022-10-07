import numpy as np
from pde_solver.system_matrix import SystemMatrix

from .system_vector import SystemVector


class LowOrderCGRightHandSide(SystemVector):
    """Right hand side of low order continuous Galerkin method (ri). To be more
    precise it is defined as following:

        ri = 1/mi*sum(d_ij*(uj-ui)-(fj-fi)*c_ij, j!=i)

    where mi denotes lumped mass, d_ij an artificial diffusion, ui the DOF
    entries, fi flux approximation and c_ij a discrete gradient.

    """

    lumped_mass: SystemVector
    artificial_diffusion: SystemMatrix
    discrete_gradient: SystemMatrix
    flux_approximation: SystemVector

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        self.artificial_diffusion.assemble(dof_vector)

        return (
            self.artificial_diffusion.dot(dof_vector)
            - self.discrete_gradient.dot(self.flux_approximation(dof_vector))
        ) / self.lumped_mass()
