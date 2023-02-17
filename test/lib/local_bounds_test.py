from test.test_helper import (
    LINEAR_LAGRANGE_SPACE,
    QUADRATIC_LAGRANGE_SPACE,
    VOLUME_MESH,
    VOLUME_SPACE,
)
from unittest import TestCase

import numpy as np
from core.finite_volume import FiniteVolumeSpace

from lib import LocalMaximum, LocalMinimum


class TestLinearLagrangeLocalBounds(TestCase):
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [[1, 1, 0, 1], [4, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 2, 1]]
    local_maximum = LocalMaximum(LINEAR_LAGRANGE_SPACE.dof_neighbours)
    local_minimum = LocalMinimum(LINEAR_LAGRANGE_SPACE.dof_neighbours)

    def test_local_maximum(self):
        for test_dofs, expected_maximum in zip(self.test_dofs, self.expected_maximum):
            self.assertListEqual(list(self.local_maximum(test_dofs)), expected_maximum)

    def test_local_minimum(self):
        for test_dofs, expected_minimum in zip(self.test_dofs, self.expected_minimum):
            self.assertListEqual(list(self.local_minimum(test_dofs)), expected_minimum)


class TestQuadraticLagrangeLocalBounds(TestLinearLagrangeLocalBounds):
    test_dofs = [np.array([1, 2, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [[2, 2, 2, 1], [4, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 1, 1]]
    local_maximum = LocalMaximum(QUADRATIC_LAGRANGE_SPACE.dof_neighbours)
    local_minimum = LocalMinimum(QUADRATIC_LAGRANGE_SPACE.dof_neighbours)


class TestVolumeSpaceNoPeriodicBoundariesLocalBounds(TestLinearLagrangeLocalBounds):
    volume_space = FiniteVolumeSpace(VOLUME_MESH)
    expected_maximum = [[1, 1, 0, 0], [2, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 2, 3]]
    local_maximum = LocalMaximum(volume_space.dof_neighbours)
    local_minimum = LocalMinimum(volume_space.dof_neighbours)


class TestSystemLocalBounds(TestCase):
    test_dof = np.array([[1, 1], [0, 2], [0, 3], [0, 4]])

    def test_local_maximum(self):
        local_maximum = LocalMaximum(VOLUME_SPACE.dof_neighbours)
        maximum = local_maximum(self.test_dof)
        expected_maximum = np.array([[1, 4], [1, 3], [0, 4], [1, 4]])
        for i in range(4):
            for j in range(2):
                self.assertEqual(maximum[i, j], expected_maximum[i, j])

    def test_local_minimum(self):
        local_minimum = LocalMinimum(VOLUME_SPACE.dof_neighbours)
        minimum = local_minimum(self.test_dof)
        expected_minimum = np.array([[0, 1], [0, 1], [0, 2], [0, 1]])
        for i in range(4):
            for j in range(2):
                self.assertEqual(minimum[i, j], expected_minimum[i, j])
