from test.test_helper import (
    LINEAR_LAGRANGE_SPACE,
    QUADRATIC_LAGRANGE_SPACE,
    VOLUME_MESH,
    VOLUME_SPACE,
)
from unittest import TestCase

import finite_volume
import numpy as np
from numpy.testing import assert_equal

from core import LocalMaximum, LocalMinimum


class TestLinearLagrangeLocalBounds(TestCase):
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [[1, 1, 0, 1], [4, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 2, 1]]
    local_maximum = LocalMaximum(LINEAR_LAGRANGE_SPACE.dof_neighbours)
    local_minimum = LocalMinimum(LINEAR_LAGRANGE_SPACE.dof_neighbours)

    def test_local_maximum(self):
        for test_dofs, expected_maximum in zip(self.test_dofs, self.expected_maximum):
            assert_equal(self.local_maximum(test_dofs), expected_maximum)

    def test_local_minimum(self):
        for test_dofs, expected_minimum in zip(self.test_dofs, self.expected_minimum):
            assert_equal(self.local_minimum(test_dofs), expected_minimum)


class TestQuadraticLagrangeLocalBounds(TestLinearLagrangeLocalBounds):
    test_dofs = [np.array([1, 2, 0, 0]), np.array([1, 2, 3, 4])]
    expected_maximum = [[2, 2, 2, 1], [4, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 1, 1]]
    local_maximum = LocalMaximum(QUADRATIC_LAGRANGE_SPACE.dof_neighbours)
    local_minimum = LocalMinimum(QUADRATIC_LAGRANGE_SPACE.dof_neighbours)


class TestVolumeSpaceNoPeriodicBoundariesLocalBounds(TestLinearLagrangeLocalBounds):
    volume_space = finite_volume.FiniteVolumeSpace(VOLUME_MESH)
    expected_maximum = [[1, 1, 0, 0], [2, 3, 4, 4]]
    expected_minimum = [[0, 0, 0, 0], [1, 1, 2, 3]]
    local_maximum = LocalMaximum(finite_volume.NeighbourIndicesMapping(4, False))
    local_minimum = LocalMinimum(finite_volume.NeighbourIndicesMapping(4, False))


class TestSystemLocalBounds(TestCase):
    test_dof = np.array([[1, 1], [0, 2], [0, 3], [0, 4]])

    def test_local_maximum(self):
        local_maximum = LocalMaximum(finite_volume.NeighbourIndicesMapping(4, True))
        maximum = local_maximum(self.test_dof)
        expected_maximum = np.array([[1, 4], [1, 3], [0, 4], [1, 4]])
        assert_equal(maximum, expected_maximum)

    def test_local_minimum(self):
        local_minimum = LocalMinimum(finite_volume.NeighbourIndicesMapping(4, True))
        minimum = local_minimum(self.test_dof)
        expected_minimum = np.array([[0, 1], [0, 1], [0, 2], [0, 1]])
        assert_equal(minimum, expected_minimum)
