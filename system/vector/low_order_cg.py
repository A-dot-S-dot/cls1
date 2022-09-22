from system.matrix.system_matrix import SystemMatrix

from .dof_vector import DOFVector
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

    _dof_vector: DOFVector

    def __init__(
        self,
        dof_vector: DOFVector,
    ):
        SystemVector.__init__(self, dof_vector.element_space)
        dof_vector.register_observer(self)

        self._dof_vector = dof_vector

    def assemble(self):
        self[:] = self.artificial_diffusion.dot(
            self._dof_vector.values
        ) - self.discrete_gradient.dot(self.flux_approximation.values)

        self[:] /= self.lumped_mass.values
