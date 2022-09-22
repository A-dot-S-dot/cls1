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
        values = abs(self._discrete_gradient.values)
        diagonal = -values.sum(axis=1)
        self._lil_values = values.tolil()
        self._lil_values.setdiag(diagonal)

        self.update_values()


class BurgersArtificialDiffusion(DiscreteUpwind):
    """Artificial diffusion operator for burgers with Rusanov Approximation.

    Exact definition is

        d_ij = max(abs(uj*a_ij), abs(ui*a_ji)) if i!=j,
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

    def assemble(self):
        values = self._discrete_gradient.values
        values = values.multiply(
            self._dof_vector.values
        )  # mulitply dof vector on each row
        values = abs(values)  # consider absolute values
        values = values.maximum(values.transpose())  # symmetrize
        diagonal = -values.sum(axis=1)  # calculate diagonal
        self._lil_values = values.tolil()
        self._lil_values.setdiag(diagonal)

        self.update_values()
