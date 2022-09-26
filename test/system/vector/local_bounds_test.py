from unittest import TestCase

import numpy as np
from system.vector.local_bounds import LocalMaximum, LocalMinimum

from ...test_helper import LINEAR_DOF_VECTOR, QUADRATIC_DOF_VECTOR


class TestLinearLocalBounds(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [np.array([1, 1, 0, 1]), np.array([4, 3, 4, 4])]
    expected_minimum = [np.array([0, 0, 0, 0]), np.array([1, 1, 2, 1])]
    local_maximum = LocalMaximum(dof_vector)
    local_minimum = LocalMinimum(dof_vector)

    def test_local_maximum(self):
        for test_dofs, expected_maximum in zip(self.test_dofs, self.expected_maximum):
            self.dof_vector.dofs = test_dofs
            self.assertListEqual(list(self.local_maximum), list(expected_maximum))

    def test_local_minimum(self):
        for test_dofs, expected_minimum in zip(self.test_dofs, self.expected_minimum):
            self.dof_vector.dofs = test_dofs
            self.assertListEqual(list(self.local_minimum), list(expected_minimum))


class TestQuadraticLocalBounds(TestLinearLocalBounds):
    dof_vector = QUADRATIC_DOF_VECTOR
    test_dofs = [np.array([1, 2, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [np.array([2, 2, 2, 1]), np.array([4, 3, 4, 4])]
    expected_minimum = [np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])]
    local_maximum = LocalMaximum(dof_vector)
    local_minimum = LocalMinimum(dof_vector)
