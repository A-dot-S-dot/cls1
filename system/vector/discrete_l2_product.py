"""Provides a classes which produces a vector containing L2 products with basis elements or its derivatives."""

from typing import List

import numpy as np
from fem import FiniteElementSpace
from fem.fast_element import FastLocalElement, FastMapping, QuadratureFastFiniteElement
from fem.lagrange import LocalLagrangeBasis
from mesh.uniform import UniformMesh

from .entry_calculator import QuadratureBasedEntryCalculator
from .system_vector import LocallyAssembledSystemVector


class AbstractBasisL2ProductEntryCalculator(QuadratureBasedEntryCalculator):
    _local_basis_elements: List[FastLocalElement]
    _determinant_derivative_affine_mapping: float

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        if not isinstance(element_space.mesh, UniformMesh):
            raise NotImplementedError(
                "Not implemented for different meshes than UniformMesh"
            )

        QuadratureBasedEntryCalculator.__init__(self, element_space, quadrature_degree)

        self._build_local_basis_elements()
        self._determinant_derivative_affine_mapping = element_space.mesh.step_length

    def _build_local_basis_elements(self):
        fast_element = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        fast_element.set_values()

        self._local_basis_elements = fast_element.local_basis_elements

    def __call__(self, simplex_index: int, local_index: int) -> float:
        right_function = self._local_basis_elements[local_index]

        node_values = np.array(
            [
                self._left_function(simplex_index, node_index)
                * right_function.value(node_index)
                for node_index in range(len(self._local_quadrature.nodes))
            ]
        )

        return self._determinant_derivative_affine_mapping * np.dot(
            self._local_quadrature.weights, node_values
        )

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        raise NotImplementedError


class AbstractBasisGradientL2ProductEntryCalculator(
    AbstractBasisL2ProductEntryCalculator
):
    _local_basis: LocalLagrangeBasis

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        AbstractBasisL2ProductEntryCalculator.__init__(
            self, element_space, quadrature_degree
        )

    def _build_local_basis_elements(self):
        fast_element = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        fast_element.set_derivatives()

        self._local_basis_elements = fast_element.local_basis_elements

    def __call__(self, simplex_index: int, local_index: int) -> float:
        right_function = self._local_basis_elements[local_index]

        node_values = np.array(
            [
                self._left_function(simplex_index, node_index)
                * right_function.derivative(node_index)
                for node_index in range(len(self._local_quadrature.nodes))
            ]
        )

        # The transformation determinant and the derivative of affine
        # transformation cancel each other out

        return np.dot(self._local_quadrature.weights, node_values)


class BasisL2ProductEntryCalculator(AbstractBasisL2ProductEntryCalculator):
    _left_function_object: FastMapping

    def __init__(
        self,
        left_function: FastMapping,
        element_space: FiniteElementSpace,
        quadrature_degree: int,
    ):
        AbstractBasisL2ProductEntryCalculator.__init__(
            self, element_space, quadrature_degree
        )

        self._left_function_object = left_function

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        return self._left_function_object.value(simplex_index, quadrature_node_index)


class BasisGradientL2ProductEntryCalculator(
    AbstractBasisGradientL2ProductEntryCalculator
):
    _left_function_object: FastMapping

    def __init__(
        self,
        left_function: FastMapping,
        element_space: FiniteElementSpace,
        quadrature_degree: int,
    ):
        AbstractBasisGradientL2ProductEntryCalculator.__init__(
            self, element_space, quadrature_degree
        )

        self._left_function_object = left_function

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        return self._left_function_object.value(simplex_index, quadrature_node_index)


class BasisL2Product(LocallyAssembledSystemVector):
    """Vector built by the L2 product between a onedimensional function f and a
    finite element basis (bi), i.e. the built vector is

    vi = (f,bi).

    """

    def __init__(
        self,
        left_function: FastMapping,
        element_space: FiniteElementSpace,
        quadrature_degree: int,
    ):
        entry_calculator = BasisL2ProductEntryCalculator(
            left_function, element_space, quadrature_degree
        )

        LocallyAssembledSystemVector.__init__(self, element_space, entry_calculator)
        self.update()


class BasisGradientL2Product(LocallyAssembledSystemVector):
    """Vector built by the L2 product between a onedimensional function f and
    the derivatives of a finite element basis (bi), i.e. the built vector is

    vi = (f,bi').

    """

    def __init__(
        self,
        left_function: FastMapping,
        element_space: FiniteElementSpace,
        quadrature_degree: int,
    ):
        entry_calculator = BasisGradientL2ProductEntryCalculator(
            left_function, element_space, quadrature_degree
        )

        LocallyAssembledSystemVector.__init__(self, element_space, entry_calculator)
        self.update()
