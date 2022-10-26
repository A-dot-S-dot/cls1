"""Provides Discrete Gradient"""
import numpy as np
from pde_solver.discretization.finite_element import LagrangeFiniteElementSpace

from .entry_calculator import QuadratureBasedEntryCalculator
from .local_matrix import LocallyAssembledSystemMatrix


class DiscreteGradientEntryCalculator(QuadratureBasedEntryCalculator):
    def __init__(self, element_space: LagrangeFiniteElementSpace):
        QuadratureBasedEntryCalculator.__init__(
            self, element_space.polynomial_degree, element_space.polynomial_degree
        )

    def __call__(
        self, cell_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        # The transformation determinant and the derivative of affine
        element_1 = self._local_basis[local_index_1]
        element_2 = self._local_basis[local_index_2]

        values = np.array(
            [
                element_1(node) * element_2.derivative(node)
                for node in self._local_quadrature.nodes
            ]
        )

        return np.dot(self._local_quadrature.weights, values)


class DiscreteGradient(LocallyAssembledSystemMatrix):
    """Discrete gradient system matrix. It's entries are phi_i * phi'_j, where {phi_i}_i
    denotes the basis of the element space.

    """

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        entry_calculator = DiscreteGradientEntryCalculator(element_space)
        LocallyAssembledSystemMatrix.__init__(self, element_space, entry_calculator)
