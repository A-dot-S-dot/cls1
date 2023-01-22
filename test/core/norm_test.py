from unittest import TestCase

import numpy as np
from numpy import sqrt

from core.mesh import Interval, UniformMesh
from core.norm import L1Norm, L2Norm, solver_spaces


class TestL2Norm(TestCase):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = L2Norm(mesh, 2)

    def test_cell_dependent_function(self):
        integrand = lambda i, x: -2.0 if x <= 0.0 else 1.0
        expected_norm = sqrt(5.0)
        self.assertAlmostEqual(self.norm(integrand), expected_norm)

    def test_scalar_norm(self):
        integrand = lambda i, x: x
        expected_norm = sqrt(2 / 3)
        self.assertAlmostEqual(self.norm(integrand), expected_norm)

    def test_multidimensional_norm(self):
        integrand = lambda i, x: np.array([x, 2 * x])
        expected_norm = np.array([sqrt(2 / 3), sqrt(8 / 3)])
        integral = self.norm(integrand)

        for i in range(2):
            self.assertAlmostEqual(integral[i], expected_norm[i])

    def test_time_evolution_multidimensional_norm(self):
        integrand = lambda i, x: np.array([[x, 2 * x], [3 * x, 4 * x], [5 * x, 6 * x]])
        expected_norm = np.array(
            [
                [sqrt(2 / 3), sqrt(8 / 3)],
                [sqrt(18 / 3), sqrt(32 / 3)],
                [sqrt(50 / 3), sqrt(72 / 3)],
            ]
        )
        integral = self.norm(integrand)

        for i in range(3):
            for j in range(2):
                self.assertAlmostEqual(integral[i, j], expected_norm[i, j])


class TestL1Norm(TestCase):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = L1Norm(mesh, 2)

    def test_cell_dependent_function(self):
        integrand = lambda i, x: -2.0 if x <= 0.0 else 1.0
        expected_norm = 3
        self.assertAlmostEqual(self.norm(integrand), expected_norm)

    def test_scalar_norm(self):
        integrand = lambda i, x: x
        expected_norm = 1.0
        self.assertAlmostEqual(self.norm(integrand), expected_norm)

    def test_multidimensional_norm(self):
        integrand = lambda i, x: np.array([x, 2 * x])
        expected_norm = np.array([1.0, 2.0])
        integral = self.norm(integrand)

        for i in range(2):
            self.assertAlmostEqual(integral[i], expected_norm[i])

    def test_time_evolution_multidimensional_norm(self):
        integrand = lambda i, x: np.array([[x, 2 * x], [3 * x, 4 * x], [5 * x, 6 * x]])
        expected_norm = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ]
        )
        integral = self.norm(integrand)

        for i in range(3):
            for j in range(2):
                self.assertAlmostEqual(integral[i, j], expected_norm[i, j])


class TestLInfNorm(TestCase):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = solver_spaces(mesh, 10)

    def test_cell_dependent_function(self):
        integrand = lambda i, x: -2.0 if x <= 0.0 else 1.0
        expected_norm = 2
        self.assertAlmostEqual(self.norm(integrand), expected_norm)

    def test_scalar_norm(self):
        integrand = lambda i, x: x
        expected_norm = 1.0
        self.assertAlmostEqual(self.norm(integrand), expected_norm)

    def test_multidimensional_norm(self):
        integrand = lambda i, x: np.array([x, 2 * x])
        expected_norm = np.array([1.0, 2.0])
        integral = self.norm(integrand)

        for i in range(2):
            self.assertAlmostEqual(integral[i], expected_norm[i])

    def test_time_evolution_multidimensional_norm(self):
        integrand = lambda i, x: np.array([[x, 2 * x], [3 * x, 4 * x], [5 * x, 6 * x]])
        expected_norm = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ]
        )
        integral = self.norm(integrand)

        for i in range(3):
            for j in range(2):
                self.assertAlmostEqual(integral[i, j], expected_norm[i, j])
