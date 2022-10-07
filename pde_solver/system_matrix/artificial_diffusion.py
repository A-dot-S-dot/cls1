import numpy as np
from pde_solver.discrete_solution import (
    DiscreteSolution,
    DiscreteSolutionObservable,
    DiscreteSolutionObserver,
)

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
        SystemMatrix.__init__(self, discrete_gradient.dimension)
        self._discrete_gradient = discrete_gradient
        self._build_values()

    def _build_values(self):
        values = abs(self._discrete_gradient())
        diagonal = -values.sum(axis=1)
        self._lil_values = values.tolil()
        self._lil_values.setdiag(diagonal)

        self.update_csr_values()


class BurgersArtificialDiffusion(DiscreteUpwind, DiscreteSolutionObserver):
    """Artificial diffusion operator for burgers with Rusanov Approximation.

    Exact definition is

        d_ij = max(abs(uj*a_ij), abs(ui*a_ji)) if i!=j,
        d_ii = - sum(d_ik, k!=i),

    where (ui) denotes the DOFs and a_ij the discrete gradient.

    Is not assembled by default.
    """

    _discrete_gradient: SystemMatrix
    _discrete_solution: DiscreteSolution

    def __init__(
        self,
        discrete_gradient: SystemMatrix,
        discrete_solution: DiscreteSolutionObservable,
    ):
        SystemMatrix.__init__(self, discrete_gradient.dimension)
        self._discrete_gradient = discrete_gradient
        self._discrete_solution = discrete_solution
        discrete_solution.register_observer(self)

    def update(self):
        self._update(self._discrete_solution.end_solution)

    def _update(self, discrete_solution: np.ndarray):
        values = self._discrete_gradient()
        values = values.multiply(discrete_solution)  # mulitply dof vector on each row
        values = abs(values)  # consider absolute values
        values = values.maximum(values.transpose())  # symmetrize
        diagonal = -values.sum(axis=1)  # calculate diagonal
        self._lil_values = values.tolil()
        self._lil_values.setdiag(diagonal)

        self.update_csr_values()

    def assemble(self, dof_vector: np.ndarray):
        if (dof_vector != self._discrete_solution.end_solution).any():
            self._update(dof_vector)
