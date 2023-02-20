from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from core.boundary import *


class TestCell(GhostCell[float]):
    def __call__(self, dof_vector: np.ndarray, time=None) -> float:
        return 0.0


class TestNodeNeighbours(TestCase):
    def test_radius_one(self):
        input = np.array([1, 2])
        conditions = BoundaryConditions(TestCell(), TestCell())
        ul, ur = conditions.get_node_neighbours(input)
        ul_expected = np.array([0, 1, 2])
        ur_expected = np.array([1, 2, 0])

        assert_equal(ul, ul_expected)
        assert_equal(ur, ur_expected)

    def test_radius_two(self):
        input = np.array([1, 2])
        conditions = BoundaryConditions(TestCell(), TestCell(), TestCell(), TestCell())
        uL, ul, ur, uR = conditions.get_node_neighbours(input, radius=2)
        uL_expected = np.array([0, 0, 1])
        ul_expected = np.array([0, 1, 2])
        ur_expected = np.array([1, 2, 0])
        uR_expected = np.array([2, 0, 0])

        assert_equal(uL, uL_expected)
        assert_equal(ul, ul_expected)
        assert_equal(ur, ur_expected)
        assert_equal(uR, uR_expected)

    def test_no_ghost_cell(self):
        input = np.array([1, 2])
        conditions = BoundaryConditions()
        ul, ur = conditions.get_node_neighbours(input)

        ul_expected = np.array([1])
        ur_expected = np.array([2])

        assert_equal(ul, ul_expected)
        assert_equal(ur, ur_expected)

    def test_periodic_condition_radius_one_and_scalar_input(self):
        input = np.array([1, 2])
        conditions = get_boundary_conditions("periodic")
        ul, ur = conditions.get_node_neighbours(input)

        assert_equal(ul, np.array([2, 1, 2]))
        assert_equal(ur, np.array([1, 2, 1]))

    def test_periodic_condition_radius_two_and_scalar_input(self):
        input = np.array([1, 2, 3, 4])
        conditions = get_boundary_conditions("periodic", radius=2)
        ul, uL, uR, ur = conditions.get_node_neighbours(input, radius=2)

        assert_equal(ul, np.array([3, 4, 1, 2, 3]))
        assert_equal(uL, np.array([4, 1, 2, 3, 4]))
        assert_equal(uR, np.array([1, 2, 3, 4, 1]))
        assert_equal(ur, np.array([2, 3, 4, 1, 2]))

    def test_periodic_condition_radius_one_and_system_input(self):
        input = np.array([[1, 1], [2, 2]])
        conditions = get_boundary_conditions("periodic")
        ul, ur = conditions.get_node_neighbours(input)

        assert_equal(ul, np.array([[2, 2], [1, 1], [2, 2]]))
        assert_equal(ur, np.array([[1, 1], [2, 2], [1, 1]]))

    def test_periodic_condition_radius_two_and_system_input(self):
        input = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        conditions = get_boundary_conditions("periodic", radius=2)
        ul, uL, uR, ur = conditions.get_node_neighbours(input, radius=2)

        assert_equal(ul, np.array([[3, 3], [4, 4], [1, 1], [2, 2], [3, 3]]))
        assert_equal(uL, np.array([[4, 4], [1, 1], [2, 2], [3, 3], [4, 4]]))
        assert_equal(uR, np.array([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1]]))
        assert_equal(ur, np.array([[2, 2], [3, 3], [4, 4], [1, 1], [2, 2]]))


class TestCellNeighbours(TestCase):
    def test_periodic_conditions_radius_1(self):
        node_values = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        conditions = get_boundary_conditions("periodic")

        values_left, values_right = conditions.get_cell_neighbours(node_values)
        expected_values_left = np.array([4.0, 1.0, 2.0, 3.0, 4.0, 1.0])
        expected_values_right = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])

        assert_equal(values_left, expected_values_left)
        assert_equal(values_right, expected_values_right)

    def test_periodic_neighbours_radius_2(self):
        node_values = np.array([4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
        conditions = get_boundary_conditions("periodic", radius=2)

        values_left, values_right = conditions.get_cell_neighbours(node_values)
        expected_values_left = np.array([3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
        expected_values_right = np.array([4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])

        assert_equal(values_left, expected_values_left)
        assert_equal(values_right, expected_values_right)

    def test_non_periodic_neighbours(self):
        node_values = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        conditions = BoundaryConditions(TestCell(), TestCell())

        values_left, values_right = conditions.get_cell_neighbours(node_values)
        expected_values_left = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 1.0])
        expected_values_right = np.array([1.0, 2.0, 3.0, 4.0, 1.0, np.nan])

        assert_equal(values_left, expected_values_left)
        assert_equal(values_right, expected_values_right)
