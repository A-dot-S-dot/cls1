import numpy as np
from pde_solver.system_matrix import SystemMatrix

from .system_vector import SystemVector
from .flux_gradient import ApproximatedFluxGradient


class LowOrderCGRightHandSide(SystemVector):
    """Right hand side of low order continuous Galerkin method (ri). To be more
    precise it is defined as following:

        ri = 1/mi*sum(d_ij*(uj-ui)-(fj-fi)*c_ij, j!=i)

    where mi denotes lumped mass, d_ij an artificial diffusion, ui the DOF
    entries, fi flux approximation and c_ij a discrete gradient.

    """

    _lumped_mass: SystemVector
    _artificial_diffusion: SystemMatrix
    _flux_gradient: ApproximatedFluxGradient

    def __init__(
        self,
        lumped_mass: SystemVector,
        artificial_diffusion: SystemMatrix,
        flux_gradient: ApproximatedFluxGradient,
    ):
        self._lumped_mass = lumped_mass
        self._artificial_diffusion = artificial_diffusion
        self._flux_gradient = flux_gradient

    def assemble(self, dof_vector: np.ndarray):
        super().assemble(dof_vector)

    def _assemble(self, dof_vector: np.ndarray):
        self._artificial_diffusion.assemble(dof_vector)

    @SystemVector.assemble_before_call
    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return (
            self._artificial_diffusion.dot(dof_vector) + self._flux_gradient(dof_vector)
        ) / self._lumped_mass()
