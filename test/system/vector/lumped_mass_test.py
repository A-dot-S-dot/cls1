from unittest import TestCase

import numpy as np
from system.vector.lumped_mass import LumpedMassVector

from ...test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE, LINEAR_MESH


class TestLinearLumpedMass(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    lumped_mass = LumpedMassVector(element_space)
    expected_mass = LINEAR_MESH.step_length * np.array([1, 1, 1, 1])

    def test_entries(self):
        for i in range(self.element_space.dimension):
            self.assertAlmostEqual(self.lumped_mass[i], self.expected_mass[i])


class TestQuadraticLumpedMass(TestLinearLumpedMass):
    element_space = QUADRATIC_LAGRANGE_SPACE
    lumped_mass = LumpedMassVector(element_space)
    expected_mass = 1 / 6 * np.array([1, 2, 1, 2])
