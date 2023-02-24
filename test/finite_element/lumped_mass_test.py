from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np

from finite_element import LumpedMassVector


class TestLinearLumpedMass(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    lumped_mass = LumpedMassVector(element_space)
    expected_mass = 1 / 4 * np.array([1, 1, 1, 1])

    def test_entries(self):
        lumped_mass = self.lumped_mass(np.empty(0))
        for i in range(len(lumped_mass)):
            self.assertAlmostEqual(
                lumped_mass[i], self.expected_mass[i], msg=f"index={i}"
            )


class TestQuadraticLumpedMass(TestLinearLumpedMass):
    element_space = QUADRATIC_LAGRANGE_SPACE
    lumped_mass = LumpedMassVector(element_space)
    expected_mass = 1 / 6 * np.array([1, 2, 1, 2])
