from system.matrix.system_matrix import SystemMatrix

from .dof_vector import DOFVector
from .group_finite_element_approximation import GroupFiniteElementApproximation
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
        self[:] = 0
        self._assemble_entries()

        self[:] /= self.lumped_mass.values

    def _assemble_entries(self):
        for element_index in range(len(self.element_space.mesh)):
            for local_index_1 in range(self.element_space.indices_per_simplex):
                i = self.element_space.get_global_index(element_index, local_index_1)

                for local_index_2 in range(self.element_space.indices_per_simplex):
                    j = self.element_space.get_global_index(
                        element_index, local_index_2
                    )

                    self._assemble_entry(i, j)

    def _assemble_entry(self, i: int, j: int):
        self[i] += (
            self.artificial_diffusion[i, j]
            * (self._dof_vector[j] - self._dof_vector[i])
            - (self.flux_approximation[j] - self.flux_approximation[i])
            * self.discrete_gradient[i, j]
        )
