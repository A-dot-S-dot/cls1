from typing import Iterator, List

import numpy as np
from scipy.interpolate import lagrange

from ..abstracts import LocalFiniteElement, LocalFiniteElementBasis


class LocalLagrangeBasis(LocalFiniteElementBasis):
    """Basis of finite elements on the standard simplex [0,1]. Each local basis
    elements can be identified with a node.

    """

    _polynomial_degree: int
    _basis_elements: List[LocalFiniteElement]
    _nodes: List[float]

    def __init__(self, polynomial_degree: int):
        self._polynomial_degree = polynomial_degree
        self._build_nodes()
        self._build_elements()

    def _build_nodes(self):
        N = self.polynomial_degree
        self._nodes = [i / N for i in range(N + 1)]

    def _build_elements(self):
        self._basis_elements = [
            self._create_local_finite_element(i) for i in range(len(self.nodes))
        ]

    def _create_local_finite_element(self, index: int) -> LocalFiniteElement:
        local_element_dof_vector = self._build_unit_vector(index)

        interpolation = lagrange(self.nodes, local_element_dof_vector)
        interpolation_derivative = interpolation.deriv()

        return LocalFiniteElement(interpolation, interpolation_derivative)

    def _build_unit_vector(self, index: int) -> np.ndarray:
        unit_vector = np.zeros(len(self.nodes))
        unit_vector[index] = 1

        return unit_vector

    def __len__(self) -> int:
        return len(self._basis_elements)

    @property
    def nodes(self) -> List[float]:
        return self._nodes

    @property
    def polynomial_degree(self) -> int:
        return self._polynomial_degree

    def __iter__(self) -> Iterator[LocalFiniteElement]:
        return iter(self._basis_elements)

    def __getitem__(self, node_index: int) -> LocalFiniteElement:
        return self._basis_elements[node_index]

    def get_element_at_node(self, node) -> LocalFiniteElement:
        return self._basis_elements[self._nodes.index(node)]
