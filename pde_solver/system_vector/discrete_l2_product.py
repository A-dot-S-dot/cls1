"""Provides a classes which produces a vector containing L2 products with basis elements or its derivatives."""

from typing import Sequence

import numpy as np
from pde_solver.discretization import FastFunction, finite_element
from pde_solver.mesh import Mesh

from .entry_calculator import QuadratureBasedEntryCalculator
from .local_vector import LocallyAssembledVector


class BasisL2ProductEntryCalculator(QuadratureBasedEntryCalculator):
    left_function: FastFunction

    _mesh: Mesh
    _local_basis_elements: Sequence[finite_element.FastLocalElement]

    def __init__(
        self,
        element_space: finite_element.LagrangeFiniteElementSpace,
        quadrature_degree: int,
    ):
        QuadratureBasedEntryCalculator.__init__(self, element_space, quadrature_degree)
        self._mesh = element_space.mesh

        self._build_local_basis_elements()

    def _build_local_basis_elements(self):
        fast_element = finite_element.QuadratureFastElement(
            self._element_space, self._local_quadrature
        )
        fast_element.set_values()

        self._local_basis_elements = fast_element.local_fast_basis

    def __call__(self, cell_index: int, local_index: int) -> float:
        right_function = self._local_basis_elements[local_index]

        node_values = np.array(
            [
                self._mesh[cell_index].length
                * self.left_function(cell_index, node_index)
                * right_function(node_index)
                for node_index in range(len(self._local_quadrature.nodes))
            ]
        )

        return np.dot(self._local_quadrature.weights, node_values)


class BasisGradientL2ProductEntryCalculator(BasisL2ProductEntryCalculator):
    _local_basis: finite_element.LocalLagrangeBasis

    def __init__(
        self,
        element_space: finite_element.LagrangeFiniteElementSpace,
        quadrature_degree: int,
    ):
        BasisL2ProductEntryCalculator.__init__(self, element_space, quadrature_degree)

    def _build_local_basis_elements(self):
        fast_element = finite_element.QuadratureFastElement(
            self._element_space, self._local_quadrature
        )
        fast_element.set_derivatives()

        self._local_basis_elements = fast_element.local_fast_basis

    def __call__(self, cell_index: int, local_index: int) -> float:
        right_function = self._local_basis_elements[local_index]

        node_values = np.array(
            [
                self.left_function(cell_index, node_index)
                * right_function.derivative(node_index)
                for node_index in range(len(self._local_quadrature.nodes))
            ]
        )

        # The transformation determinant and the derivative of affine
        # transformation cancel each other out

        return np.dot(self._local_quadrature.weights, node_values)


class BasisL2Product(LocallyAssembledVector):
    """Vector built by the L2 product between a onedimensional function f and a
    finite element basis (bi), i.e. the built vector is

    vi = (f,bi).

    """

    def __init__(
        self,
        left_function: FastFunction,
        element_space: finite_element.LagrangeFiniteElementSpace,
        quadrature_degree: int,
    ):
        entry_calculator = BasisL2ProductEntryCalculator(
            element_space, quadrature_degree
        )
        entry_calculator.left_function = left_function

        LocallyAssembledVector.__init__(self, element_space, entry_calculator)


class BasisGradientL2Product(LocallyAssembledVector):
    """Vector built by the L2 product between a onedimensional function f and
    the derivatives of a finite element basis (bi), i.e. the built vector is

    vi = (f,bi').

    """

    def __init__(
        self,
        left_function: FastFunction,
        element_space: finite_element.LagrangeFiniteElementSpace,
        quadrature_degree: int,
    ):
        entry_calculator = BasisGradientL2ProductEntryCalculator(
            element_space, quadrature_degree
        )
        entry_calculator.left_function = left_function

        LocallyAssembledVector.__init__(self, element_space, entry_calculator)
