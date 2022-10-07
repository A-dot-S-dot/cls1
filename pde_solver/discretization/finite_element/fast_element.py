from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from custom_type import ScalarFunction
from pde_solver.discretization.local_lagrange import (
    LOCAL_LAGRANGE_BASIS,
    LocalLagrangeElement,
)
from pde_solver.mesh import AffineTransformation, Mesh
from pde_solver.quadrature import LocalElementQuadrature
from pde_solver.solver_space import LagrangeFiniteElementSpace


class FastMapping(ABC):
    """Returns a value or derivative for certain simplex indices of a mesh and a
    point which is also accessed by an index.

    """

    @abstractmethod
    def __call__(self, cell_index: int, local_index: int) -> float:
        ...

    @abstractmethod
    def derivative(self, cell_index: int, local_index: int) -> float:
        ...


class FastFunction(FastMapping):
    _function: ScalarFunction
    _derivative: ScalarFunction
    _mesh: Mesh
    _local_points: Sequence[float]

    _affine_transformation: AffineTransformation

    def __init__(
        self,
        function: ScalarFunction,
        derivative: ScalarFunction,
        mesh: Mesh,
        local_points: Sequence[float],
    ):
        self._function = function
        self._derivative = derivative
        self._mesh = mesh
        self._local_points = local_points

        self._affine_transformation = AffineTransformation()

    def __call__(self, cell_index: int, local_index: int) -> float:
        cell = self._mesh[cell_index]
        local_point = self._local_points[local_index]

        return self._function(self._affine_transformation(local_point, cell))

    def derivative(self, simplex_index: int, local_index: int) -> float:
        simplex = self._mesh[simplex_index]
        local_point = self._local_points[local_index]

        return self._derivative(self._affine_transformation(local_point, simplex))


class FastLocalElement:
    """This element is made for better performance while calculating values and
    derivatives of local Lagrange elements.

    Invoke 'set_values' and 'set_derivatives' for adding values and derivatives.

    The desired value can be accessed via 'value' or 'derivative'.

    """

    _local_element: LocalLagrangeElement
    _values: Sequence[float]
    _derivatives: Sequence[float]

    def __init__(self, local_element: LocalLagrangeElement):
        self._local_element = local_element

    def set_values(self, *local_points: float):
        self._values = [self._local_element(point) for point in local_points]

    def set_derivatives(self, *local_points: float):
        self._derivatives = [
            self._local_element.derivative(point) for point in local_points
        ]

    def __call__(self, local_index: int) -> float:
        return self._values[local_index]

    def derivative(self, local_index: int) -> float:
        return self._derivatives[local_index]


class FastFiniteElement(FastMapping):
    """This element is made for better performance while calculating values and
    derivatives of Lagrange elements.

    Essentially it calculates finite elements on all points which are related to
    certain local points on the standard simplex. This is done by calculating
    the values (and if needed derivatives) in the preprocessing step by invoking
    'set_values' and 'set_derivatives'.

    The desired value can be accessed via 'value_on_simplex' or
    'derivative_on_simplex'.

    """

    dof_vector: np.ndarray
    local_fast_basis: Sequence[FastLocalElement]

    _element_space: LagrangeFiniteElementSpace
    _affine_transformation: AffineTransformation

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        self._element_space = element_space
        self._affine_transformation = AffineTransformation()
        self._build_fast_basis()

    def _build_fast_basis(self):
        local_basis = LOCAL_LAGRANGE_BASIS[self._element_space.polynomial_degree]
        self.local_fast_basis = [
            FastLocalElement(basis_element) for basis_element in local_basis
        ]

    def set_values(self, *local_points: float):
        for element in self.local_fast_basis:
            element.set_values(*local_points)

    def set_derivatives(self, *local_points: float):
        for element in self.local_fast_basis:
            element.set_derivatives(*local_points)

    def __call__(self, cell_index: int, local_index: int) -> float:
        local_dof_vector = self._build_local_dof_vector(cell_index)
        local_basis_elements_values = [
            element(local_index) for element in self.local_fast_basis
        ]
        return np.dot(local_dof_vector, local_basis_elements_values)

    def _build_local_dof_vector(self, cell_index: int) -> np.ndarray:
        local_dof_vector = np.zeros(self._element_space.polynomial_degree + 1)

        for local_index in range(self._element_space.polynomial_degree + 1):
            global_index = self._element_space.global_index(cell_index, local_index)
            local_dof_vector[local_index] = self.dof_vector[global_index]

        return local_dof_vector

    def derivative(self, cell_index: int, local_index: int) -> float:
        local_dof_vector = self._build_local_dof_vector(cell_index)
        local_basis_elements_derivatives = [
            element.derivative(local_index) for element in self.local_fast_basis
        ]

        cell = self._element_space.mesh[cell_index]
        distortion = self._affine_transformation.inverse_derivative(cell)

        return distortion * np.dot(local_dof_vector, local_basis_elements_derivatives)


class QuadratureFastFiniteElement(FastFiniteElement):
    """Element for fast calculation of values and derivatives in quadrature
    nodes.

    """

    _quadrature_nodes: Sequence[float]

    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        FastFiniteElement.__init__(self, element_space)

        self._quadrature_nodes = local_quadrature.nodes

    def set_values(self):
        super().set_values(*self._quadrature_nodes)

    def set_derivatives(self):
        super().set_derivatives(*self._quadrature_nodes)


class AnchorNodesFastFiniteElement(FastFiniteElement):
    """Element for fast calculation of values and derivatives in anchor
    nodes of local Lagrange elements.

    """

    _anchor_nodes: Sequence[float]

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        FastFiniteElement.__init__(self, element_space)

        self._anchor_nodes = LOCAL_LAGRANGE_BASIS[element_space.polynomial_degree].nodes

    def set_values(self):
        super().set_values(*self._anchor_nodes)

    def set_derivatives(self):
        super().set_derivatives(*self._anchor_nodes)
