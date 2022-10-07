from typing import Iterator, List

import numpy as np
from custom_type import ScalarFunction
from scipy.interpolate import lagrange


class LocalLagrangeElement:
    """Finite element basis element on the standard simplex."""

    _call_method: ScalarFunction
    _derivative: ScalarFunction

    def __init__(
        self,
        call_method: ScalarFunction,
        derivative: ScalarFunction,
    ):
        self._call_method = call_method
        self._derivative = derivative

    def __call__(self, point: float) -> float:
        return self._call_method(point)

    def derivative(self, point: float) -> float:
        return self._derivative(point)


class LocalLagrangeBasis:
    """Basis of finite elements on the standard simplex [0,1]. Each local basis
    elements can be identified with a node.

    """

    polynomial_degree: int
    nodes: List[float]

    _basis_elements: List[LocalLagrangeElement]

    def __init__(self, polynomial_degree: int):
        self.polynomial_degree = polynomial_degree
        self._build_nodes()
        self._build_elements()

    def _build_nodes(self):
        N = self.polynomial_degree
        self.nodes = [i / N for i in range(N + 1)]

    def _build_elements(self):
        self._basis_elements = [
            self._create_local_finite_element(i) for i in range(len(self.nodes))
        ]

    def _create_local_finite_element(self, index: int) -> LocalLagrangeElement:
        local_element_dof_vector = self._build_unit_vector(index)

        interpolation = lagrange(self.nodes, local_element_dof_vector)
        interpolation_derivative = interpolation.deriv()

        return LocalLagrangeElement(interpolation, interpolation_derivative)

    def _build_unit_vector(self, index: int) -> np.ndarray:
        unit_vector = np.zeros(len(self.nodes))
        unit_vector[index] = 1

        return unit_vector

    def __len__(self) -> int:
        return len(self._basis_elements)

    def __iter__(self) -> Iterator[LocalLagrangeElement]:
        return iter(self._basis_elements)

    def __getitem__(self, node_index: int) -> LocalLagrangeElement:
        return self._basis_elements[node_index]

    def get_element_at_node(self, node) -> LocalLagrangeElement:
        return self._basis_elements[self.nodes.index(node)]


class LocalLagrangeBasisContainer:
    _basis = {}

    def __getitem__(self, polynomial_degree: int) -> LocalLagrangeBasis:
        try:
            return self._basis[polynomial_degree]
        except KeyError:
            self._basis[polynomial_degree] = LocalLagrangeBasis(polynomial_degree)
            return self._basis[polynomial_degree]


LOCAL_LAGRANGE_BASIS = LocalLagrangeBasisContainer()
