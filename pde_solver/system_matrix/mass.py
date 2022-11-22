import numpy as np
from pde_solver.mesh import Mesh
from pde_solver.discretization.finite_element import LagrangeFiniteElementSpace

from .entry_calculator import QuadratureBasedEntryCalculator
from .local_matrix import LocallyAssembledSystemMatrix


class MassEntryCalculator(QuadratureBasedEntryCalculator):
    _mesh: Mesh

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        QuadratureBasedEntryCalculator.__init__(
            self, element_space.polynomial_degree, element_space.polynomial_degree + 1
        )
        self._mesh = element_space.mesh

    def __call__(
        self, cell_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        element_1 = self._local_basis[local_index_1]
        element_2 = self._local_basis[local_index_2]

        values = np.array(
            [
                self._mesh[cell_index].length * element_1(node) * element_2(node)
                for node in self._local_quadrature.nodes
            ]
        )

        return np.dot(self._local_quadrature.weights, values)


class MassMatrix(LocallyAssembledSystemMatrix):
    """Mass system matrix. It's entries are phi_i * phi_j, where {phi_i}_i
    denotes the basis of the element space.

    """

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        entry_calculator = MassEntryCalculator(element_space)
        LocallyAssembledSystemMatrix.__init__(self, element_space, entry_calculator)
