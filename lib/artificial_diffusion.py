import numpy as np
from core import SystemMatrix
from core.finite_element import LagrangeSpace

from .discrete_gradient import DiscreteGradient


class DiscreteUpwind(SystemMatrix):
    """An artificial diffusion operator for linear advection depending on a
    discrete advection a_{ij}.

    Exact definition is

        d_ij = max(a_ij, a_ji) if i!=j,
        d_ii = - sum(d_ik, k!=i).

    """

    _discrete_gradient: SystemMatrix

    def __init__(self, element_space: LagrangeSpace):
        self._discrete_gradient = DiscreteGradient(element_space)
        SystemMatrix.__init__(self, self._discrete_gradient.dimension)
        self._build_values()

    def _build_values(self):
        values = abs(self._discrete_gradient())
        diagonal = -values.sum(axis=1)
        self._lil_values = values.tolil()
        self._lil_values.setdiag(diagonal)

        self.update_csr_values()


class BurgersArtificialDiffusion(DiscreteUpwind):
    """Artificial diffusion operator for burgers with Rusanov Approximation.

    Exact definition is

        d_ij = max(abs(uj*a_ij), abs(ui*a_ji)) if i!=j,
        d_ii = - sum(d_ik, k!=i),

    where (ui) denotes the DOFs and a_ij the discrete gradient.

    Is not assembled by default.
    """

    _discrete_gradient: SystemMatrix

    def __init__(self, element_space: LagrangeSpace):
        self._discrete_gradient = DiscreteGradient(element_space)
        SystemMatrix.__init__(self, self._discrete_gradient.dimension)

    def assemble(self, dof_vector: np.ndarray):
        values = self._discrete_gradient()
        values = values.multiply(dof_vector)  # mulitply dof vector on each row
        values = abs(values)  # consider absolute values
        values = values.maximum(values.transpose())  # symmetrize
        diagonal = -values.sum(axis=1)  # calculate diagonal
        self._lil_values = values.tolil()
        self._lil_values.setdiag(diagonal)

        self.update_csr_values()


def build_artificial_diffusion(
    problem: str, element_space: LagrangeSpace
) -> SystemMatrix:
    diffusions = {
        "advection": DiscreteUpwind,
        "burgers": BurgersArtificialDiffusion,
    }
    return diffusions[problem](element_space)
