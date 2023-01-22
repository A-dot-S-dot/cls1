from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np

from lib import LocalMaximum, LocalMinimum


class TestLinearLocalBounds(TestCase):
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [[1, 1, 0, 1], [4, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 2, 1]]
    local_maximum = LocalMaximum(LINEAR_LAGRANGE_SPACE)
    local_minimum = LocalMinimum(LINEAR_LAGRANGE_SPACE)

    def test_local_maximum(self):
        for test_dofs, expected_maximum in zip(self.test_dofs, self.expected_maximum):
            self.assertListEqual(list(self.local_maximum(test_dofs)), expected_maximum)

    def test_local_minimum(self):
        for test_dofs, expected_minimum in zip(self.test_dofs, self.expected_minimum):
            self.assertListEqual(list(self.local_minimum(test_dofs)), expected_minimum)


class TestQuadraticLocalBounds(TestLinearLocalBounds):
    test_dofs = [np.array([1, 2, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [[2, 2, 2, 1], [4, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 1, 1]]
    local_maximum = LocalMaximum(QUADRATIC_LAGRANGE_SPACE)
    local_minimum = LocalMinimum(QUADRATIC_LAGRANGE_SPACE)
