from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from core.boundary import *


class TestCell(GhostCell[float]):
    def __call__(self, dof_vector: np.ndarray, time=None) -> float:
        return 0.0


class TestNeighboursWithGhostCells(TestCase):
    def test_radius_one(self):
        input = np.array([1, 2])
        neighbours = NodeNeighbours(TestCell(), TestCell())
        ul, ur = neighbours(input)
        ul_expected = np.array([0, 1, 2])
        ur_expected = np.array([1, 2, 0])

        assert_equal(ul, ul_expected)
        assert_equal(ur, ur_expected)

    def test_radius_two(self):
        input = np.array([1, 2])
        neighbours = NodeNeighbours(TestCell(), TestCell(), TestCell(), TestCell())
        uL, ul, ur, uR = neighbours(input)
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
        neighbours = NodeNeighbours()
        ul, ur = neighbours(input)

        ul_expected = np.array([1])
        ur_expected = np.array([2])

        assert_equal(ul, ul_expected)
        assert_equal(ur, ur_expected)


class TestNeighboursWithPeriodicBoundary(TestCase):
    def test_radius_one_and_scalar_input(self):
        input = np.array([1, 2])
        neighbours = build_node_neighbours("periodic")
        ul, ur = neighbours(input)

        assert_equal(ul, np.array([2, 1, 2]))
        assert_equal(ur, np.array([1, 2, 1]))

    def test_radius_two_and_scalar_input(self):
        input = np.array([1, 2, 3, 4])
        neighbours = build_node_neighbours("periodic", 2)
        ul, uL, uR, ur = neighbours(input)

        assert_equal(ul, np.array([3, 4, 1, 2, 3]))
        assert_equal(uL, np.array([4, 1, 2, 3, 4]))
        assert_equal(uR, np.array([1, 2, 3, 4, 1]))
        assert_equal(ur, np.array([2, 3, 4, 1, 2]))

    def test_radius_one_and_system_input(self):
        input = np.array([[1, 1], [2, 2]])
        neighbours = build_node_neighbours("periodic")
        ul, ur = neighbours(input)

        assert_equal(ul, np.array([[2, 2], [1, 1], [2, 2]]))
        assert_equal(ur, np.array([[1, 1], [2, 2], [1, 1]]))

    def test_radius_two_and_system_input(self):
        input = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        neighbours = build_node_neighbours("periodic", 2)
        ul, uL, uR, ur = neighbours(input)

        assert_equal(ul, np.array([[3, 3], [4, 4], [1, 1], [2, 2], [3, 3]]))
        assert_equal(uL, np.array([[4, 4], [1, 1], [2, 2], [3, 3], [4, 4]]))
        assert_equal(uR, np.array([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1]]))
        assert_equal(ur, np.array([[2, 2], [3, 3], [4, 4], [1, 1], [2, 2]]))
