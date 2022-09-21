import numpy as np
from system.vector import DOFVector

from .system_matrix import SystemMatrix


class DiscreteUpwind(SystemMatrix):
    """An artificial diffusion operator for linear advection depending on a
    discrete advection a_{ij}.

    Exact definition is

        d_ij = max(a_ij, a_ji) if i!=j,
        d_ii = - sum(d_ik, k!=i).

    """

    _discrete_gradient: SystemMatrix

    def __init__(self, discrete_gradient: SystemMatrix):
        SystemMatrix.__init__(self, discrete_gradient.element_space)
        self._discrete_gradient = discrete_gradient
        self.assemble()

    def assemble(self):
        # Since the matrix is symmetric it is sufficient to calculate the
        # entries for j<=i. But the entry (i,i) depends on all entries (i,j)
        # with j!=i. Therefore, we iterate over j<i.
        for simplex_index in range(len(self.element_space.mesh)):
            for local_index_1 in range(self.element_space.indices_per_simplex):
                i = self.element_space.get_global_index(simplex_index, local_index_1)
                for local_index_2 in range(self.element_space.indices_per_simplex):
                    j = self.element_space.get_global_index(
                        simplex_index, local_index_2
                    )

                    if j > i:
                        self._set_entry(i, j)

    def _set_entry(self, i: int, j: int):
        entry = self._get_entry(i, j)

        self[i, j] = self[j, i] = entry
        self[i, i] -= entry
        self[j, j] -= entry

    def _get_entry(self, i: int, j: int) -> float:
        return max(self._discrete_gradient[i, j], self._discrete_gradient[j, i])


class BurgersArtificialDiffusion(DiscreteUpwind):
    """Artificial diffusion operator for burgers with Rusanov Approximation.

    Exact definition is

        d_ij = max(uj*a_ij, ui*a_ji) if i!=j,
        d_ii = - sum(d_ik, k!=i),

    where (ui) denotes the DOFs and a_ij the discrete gradient.

    Is not assembled by default.
    """

    _dof_vector: DOFVector
    _discrete_gradient: SystemMatrix

    def __init__(
        self,
        dof_vector: DOFVector,
        discrete_gradient: SystemMatrix,
    ):
        SystemMatrix.__init__(self, dof_vector.element_space)
        self._dof_vector = dof_vector
        self._discrete_gradient = discrete_gradient

        dof_vector.register_observer(self)

    def _get_entry(self, i: int, j: int) -> float:
        return max(
            self._dof_vector[i] * self._discrete_gradient[i, j],
            self._dof_vector[j] * self._discrete_gradient[j, i],
        )
