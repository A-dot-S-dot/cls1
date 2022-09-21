from typing import List
from unittest import TestCase

import numpy as np
from fem.global_element import GlobalFiniteElement
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh import Interval
from mesh.uniform import UniformMesh

from ...test_helper import PiecewiseLagrangeInterpolation


class TestLinearLagrangeFiniteElementSpace(TestCase):
    polynomial_degree = 1
    element_number = 2
    indices_per_simplex = 2
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, element_number)
    element_space = LagrangeFiniteElementSpace(mesh, polynomial_degree)
    expected_basis_nodes = [0, 0.5]
    basis_coefficients = [(1, 0), (0, 1)]
    test_basis: List[PiecewiseLagrangeInterpolation]
    test_points = np.linspace(interval.a, interval.b)

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

        self._build_test_basis()

    def _build_test_basis(self):
        element_1 = PiecewiseLagrangeInterpolation()
        element_1.add_piecewise_polynomial((0, 0.5), (1, 0), Interval(0, 0.5))
        element_1.add_piecewise_polynomial((0.5, 1), (0, 1), Interval(0.5, 1))

        element_2 = PiecewiseLagrangeInterpolation()
        element_2.add_piecewise_polynomial((0, 0.5), (0, 1), Interval(0, 0.5))
        element_2.add_piecewise_polynomial((0.5, 1), (1, 0), Interval(0.5, 1))

        self.test_basis = [element_1, element_2]

    def test_polynomial_degree(self):
        self.assertEqual(self.element_space.polynomial_degree, self.polynomial_degree)

    def test_dimension(self):
        self.assertEqual(
            self.element_space.dimension,
            self.element_number * self.polynomial_degree,
        )

    def test_indices_per_simplex(self):
        self.assertEqual(
            self.element_space.indices_per_simplex, self.indices_per_simplex
        )

    def test_basis_nodes(self):
        for basis_node, expected_basis_node in zip(
            self.element_space.basis_nodes, self.expected_basis_nodes
        ):
            self.assertEqual(basis_node, expected_basis_node)

    def test_basis_values(self):
        for basis_coefficients, test_element in zip(
            self.basis_coefficients, self.test_basis
        ):
            for point in self.test_points:
                self.assertAlmostEqual(
                    self.element_space.get_value(point, basis_coefficients),
                    test_element(point),
                )

    def test_basis_derivative(self):
        for basis_coefficients, test_element in zip(
            self.basis_coefficients, self.test_basis
        ):
            for point in self.test_points:
                if not np.isnan(test_element.derivative(point)):
                    self.assertAlmostEqual(
                        self.element_space.get_derivative(point, basis_coefficients),
                        test_element.derivative(point),
                        msg=f"basis_dof={basis_coefficients}, point={point}",
                    )
                else:
                    self.assertIs(
                        self.element_space.get_derivative(point, basis_coefficients),
                        np.nan,
                    )

    def test_not_interpolable_error(self):
        test_function = lambda x: x
        self.assertRaises(ValueError, self.element_space.interpolate, test_function)

    def test_interpolate(self):
        test_function = lambda _: 5
        interpolator = GlobalFiniteElement(
            self.element_space, self.element_space.interpolate(test_function)
        )

        for point in self.test_points:
            self.assertAlmostEqual(interpolator(point), test_function(point))

    def test_is_dof_vector(self):
        dof_vector = np.array((1, 2))
        not_dof_vector_1 = np.array([1])
        not_dof_vector_2 = np.array((1, 2, 3, 4, 5))

        self.assertTrue(self.element_space.is_dof_vector(dof_vector))
        self.assertFalse(self.element_space.is_dof_vector(not_dof_vector_1))
        self.assertFalse(self.element_space.is_dof_vector(not_dof_vector_2))


class TestQuadraticLagrangeFiniteElementSpace(TestLinearLagrangeFiniteElementSpace):
    polynomial_degree = 2
    element_number = 2
    indices_per_simplex = 3
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, element_number)
    element_space = LagrangeFiniteElementSpace(mesh, polynomial_degree)
    expected_basis_nodes = [0, 0.25, 0.5, 0.75]
    basis_coefficients = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    test_basis: List[PiecewiseLagrangeInterpolation]
    test_points = np.linspace(interval.a, interval.b)

    def _build_test_basis(self):
        element_1 = PiecewiseLagrangeInterpolation()
        element_1.add_piecewise_polynomial((0, 0.25, 0.5), (1, 0, 0), Interval(0, 0.5))
        element_1.add_piecewise_polynomial((0.5, 0.75, 1), (0, 0, 1), Interval(0.5, 1))

        element_2 = PiecewiseLagrangeInterpolation()
        element_2.add_piecewise_polynomial((0, 0.25, 0.5), (0, 1, 0), Interval(0, 0.5))
        element_2.add_piecewise_polynomial((0.5, 0.75, 1), (0, 0, 0), Interval(0.5, 1))

        element_3 = PiecewiseLagrangeInterpolation()
        element_3.add_piecewise_polynomial((0, 0.25, 0.5), (0, 0, 1), Interval(0, 0.5))
        element_3.add_piecewise_polynomial((0.5, 0.75, 1), (1, 0, 0), Interval(0.5, 1))

        element_4 = PiecewiseLagrangeInterpolation()
        element_4.add_piecewise_polynomial((0, 0.25, 0.5), (0, 0, 0), Interval(0, 0.5))
        element_4.add_piecewise_polynomial((0.5, 0.75, 1), (0, 1, 0), Interval(0.5, 1))

        self.test_basis = [element_1, element_2, element_3, element_4]

    def test_not_interpolable_error(self):
        test_function = lambda x: x**2
        self.assertRaises(ValueError, self.element_space.interpolate, test_function)

    def test_interpolate(self):
        test_function = lambda x: x * (x - 1)
        interpolator = GlobalFiniteElement(
            self.element_space, self.element_space.interpolate(test_function)
        )

        for point in self.test_points:
            self.assertAlmostEqual(interpolator(point), test_function(point))

    def test_is_dof_vector(self):
        dof_vector = np.array((1, 2, 3, 4))
        not_dof_vector_1 = np.array((1, 2, 3))
        not_dof_vector_2 = np.array((1, 2, 3, 4, 5))

        self.assertTrue(self.element_space.is_dof_vector(dof_vector))
        self.assertFalse(self.element_space.is_dof_vector(not_dof_vector_1))
        self.assertFalse(self.element_space.is_dof_vector(not_dof_vector_2))
