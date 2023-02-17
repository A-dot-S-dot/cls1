from typing import List
from unittest import TestCase

from numpy import sqrt

from core.mesh import Interval
from core.quadrature import *


class TestGaussianQuadrature(TestCase):
    def test_value_error(self):
        self.assertRaises(AssertionError, SpecificGaussianQuadrature, 0)
        self.assertRaises(AssertionError, SpecificGaussianQuadrature, -2)

    def test_nodes_and_weights(self):
        self._test_nodes_and_weights(1, [0], [2])
        self._test_nodes_and_weights(2, [-1 / sqrt(3), 1 / sqrt(3)], [1, 1])
        self._test_nodes_and_weights(
            3, [-sqrt(3 / 5), 0, sqrt(3 / 5)], [5 / 9, 8 / 9, 5 / 9]
        )

    def _test_nodes_and_weights(
        self, nodes_number: int, nodes: List[float], weights: List[float]
    ):
        quadrature = SpecificGaussianQuadrature(nodes_number)
        for exact_node, exact_weight, node, weight in zip(
            nodes, weights, quadrature.nodes, quadrature.weights
        ):
            self.assertAlmostEqual(exact_node, node)
            self.assertAlmostEqual(exact_weight, weight)

    def test_domain(self):
        quadrature = SpecificGaussianQuadrature(1)
        self.assertEqual(quadrature.domain, Interval(-1, 1))

    def test_integration(self):
        quadrature = SpecificGaussianQuadrature(3)
        test_functions_integrals = [
            (lambda x: x**5, 0),
            (lambda x: x + 1, 2),
        ]

        for f, integral in test_functions_integrals:
            self.assertAlmostEqual(quadrature.integrate(f), integral)


class TestGaussianQuadratureGeneralized(TestCase):
    interval = Interval(0, 1)

    def test_nodes_and_weights(self):
        self._test_nodes_and_weights(1, [0.5], [1])
        self._test_nodes_and_weights(
            2, [-1 / sqrt(12) + 1 / 2, 1 / sqrt(12) + 1 / 2], [1 / 2, 1 / 2]
        )
        self._test_nodes_and_weights(
            3,
            [-sqrt(3 / 20) + 1 / 2, 1 / 2, sqrt(3 / 20) + 1 / 2],
            [5 / 18, 4 / 9, 5 / 18],
        )

    def _test_nodes_and_weights(
        self, nodes_number: int, nodes: List[float], weights: List[float]
    ):
        quadrature = GaussianQuadrature(nodes_number, self.interval)
        for exact_node, exact_weight, node, weight in zip(
            nodes, weights, quadrature.nodes, quadrature.weights
        ):
            self.assertAlmostEqual(exact_node, node)
            self.assertAlmostEqual(exact_weight, weight)

    def test_integration(self):
        quadrature = GaussianQuadrature(3, self.interval)
        test_functions_integrals = [
            (lambda x: x**5, 1 / 6),
            (lambda x: x + 1, 3 / 2),
        ]

        for f, integral in test_functions_integrals:
            self.assertAlmostEqual(quadrature.integrate(f), integral)


class TestLocalElementQuadrature(TestCase):
    def test_local_quadrature(self):
        local_element_quadrature = LocalElementQuadrature(2)

        functions = [
            lambda _: 1,
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
        ]

        integrals = [1, 1 / 2, 1 / 3, 1 / 4]

        for f, integral in zip(functions, integrals):
            self.assertAlmostEqual(local_element_quadrature.integrate(f), integral)
