from unittest import TestCase

import numpy as np
from pde_solver.discretization.local_lagrange import (
    LOCAL_LAGRANGE_BASIS,
    LocalLagrangeBasis,
    LocalLagrangeElement,
)

from test.test_helper import discrete_derivative


class TestLocalLagrangeBasis(TestCase):
    test_polynomial_degrees = [p + 1 for p in range(8)]
    local_bases = LOCAL_LAGRANGE_BASIS

    def test_nodes(self):
        expected_nodes = [
            [0, 1],
            [0, 1 / 2, 1],
            [0, 1 / 3, 2 / 3, 1],
        ]

        for p, expected_basis_nodes in enumerate(expected_nodes):
            basis = self.local_bases[p + 1]
            for node, expected_node in zip(basis.nodes, expected_basis_nodes):
                self.assertEqual(node, expected_node)

    def test_delta_property(self):
        for p in self.test_polynomial_degrees:
            basis = LOCAL_LAGRANGE_BASIS[p]
            self._test_delta_property_for_basis(basis)

    def _test_delta_property_for_basis(self, basis: LocalLagrangeBasis):
        for i in range(len(basis.nodes)):
            self._compare_basis_element_with_basis(i, basis)

    def _compare_basis_element_with_basis(
        self, basis_index: int, basis: LocalLagrangeBasis
    ):
        node_i = basis.nodes[basis_index]
        basis_element = basis.get_element_at_node(node_i)

        for j, node_j in enumerate(basis.nodes):
            self.assertAlmostEqual(
                basis_element(node_j),
                float(node_i == node_j),
                msg=f"p={basis.polynomial_degree}, basis_index={basis_index}, node_index={j}",
            )

    def test_len(self):
        basis = self.local_bases[2]
        self.assertEqual(len(basis), 3)

    def test_derivative(self):
        for p in range(1, 4):
            basis = self.local_bases[p]
            self._test_derivative_for_basis(basis)

    def _test_derivative_for_basis(self, basis: LocalLagrangeBasis):
        for element_index, element in enumerate(basis):
            self._test_derivative_for_element(element, element_index, basis)

    def _test_derivative_for_element(
        self,
        element: LocalLagrangeElement,
        element_index: int,
        basis: LocalLagrangeBasis,
    ):
        for x in np.linspace(0, 1):
            self.assertAlmostEqual(
                element.derivative(x),
                discrete_derivative(element, x),
                msg=f"p={basis.polynomial_degree} , element={element_index}, point={x}",
                delta=1e-7,
            )
