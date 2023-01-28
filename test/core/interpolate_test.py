from itertools import product
from test.test_helper import LINEAR_LAGRANGE_SPACE, LINEAR_MESH, VOLUME_SPACE
from unittest import TestCase

import numpy as np

from core import interpolate


class TestNodeInterpolator(TestCase):
    def test_interpolator(self):
        f = lambda _: 1
        interpolator = interpolate.NodeValuesInterpolator(
            *LINEAR_LAGRANGE_SPACE.basis_nodes
        )
        interpolation = interpolator.interpolate(f)
        expected_interpolation = [1, 1, 1, 1]

        self.assertListEqual(list(interpolation), expected_interpolation)


class TestCellInterpolator(TestCase):
    def test_interpolator(self):
        f = lambda _: 1
        interpolator = interpolate.CellAverageInterpolator(LINEAR_MESH, 1)
        interpolation = interpolator.interpolate(f)
        expected_interpolation = [1, 1, 1, 1]

        self.assertListEqual(list(interpolation), expected_interpolation)


class TestTemporalInterpolation(TestCase):
    def test_interpolator(self):
        old_time = np.array([0, 1])
        values = np.array([[[0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]])
        interpolation_times = np.array([0.5])

        interpolator = interpolate.TemporalInterpolator()
        interpolated_values = interpolator(old_time, values, interpolation_times)

        self.assertTupleEqual(interpolated_values.shape, (1, 3, 2))
        self.assertListEqual(list(interpolated_values.flatten()), 6 * [0.5])


class TestSpatialInterpolation(TestCase):
    def test_scalar_interpolator(self):
        old_grid = np.array([0.25, 0.75])
        values = np.array([[1.0, 3.0], [5.0, 7.0]])
        new_grid = np.array([1 / 8, 3 / 8, 5 / 8, 7 / 8])
        expected_interpolation = np.array([[1.0, 1.5, 2.5, 3.0], [5.0, 5.5, 6.5, 7.0]])

        interpolator = interpolate.SpatialInterpolator()
        interpolation = interpolator(old_grid, values, new_grid)

        self.assertTupleEqual(interpolation.shape, (2, 4))

        for i, j in product(range(2), range(4)):
            self.assertAlmostEqual(interpolation[i, j], expected_interpolation[i, j])

    def test_system_interpolator(self):
        old_grid = np.array([0.25, 0.75])
        values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        new_grid = np.array([1 / 8, 3 / 8, 5 / 8, 7 / 8])
        expected_interpolation = np.array(
            [
                [[1.0, 2.0], [1.5, 2.5], [2.5, 3.5], [3.0, 4.0]],
                [[5.0, 6.0], [5.5, 6.5], [6.5, 7.5], [7.0, 8.0]],
            ]
        )

        interpolator = interpolate.SpatialInterpolator()
        interpolation = interpolator(old_grid, values, new_grid)

        self.assertTupleEqual(interpolation.shape, (2, 4, 2))

        for i, j, k in product(range(2), range(4), range(2)):
            self.assertAlmostEqual(
                interpolation[i, j, k], expected_interpolation[i, j, k]
            )
