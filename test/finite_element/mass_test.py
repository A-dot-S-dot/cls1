from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np

from finite_element import MassMatrix


class TestLinearMass(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    mass = MassMatrix(element_space)
    expected_mass = (
        1 / 24 * np.array([[4, 1, 0, 1], [1, 4, 1, 0], [0, 1, 4, 1], [1, 0, 1, 4]])
    )

    def test_entries(self):
        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertAlmostEqual(self.mass[i, j], self.expected_mass[i, j])


class TestQuadraticMass(TestLinearMass):
    element_space = QUADRATIC_LAGRANGE_SPACE
    mass = MassMatrix(element_space)
    expected_mass = (
        1 / 30 * np.array([[4, 1, -1, 1], [1, 8, 1, 0], [-1, 1, 4, 1], [1, 0, 1, 8]])
    )
