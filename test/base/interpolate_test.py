from test.test_helper import LINEAR_LAGRANGE_SPACE, LINEAR_MESH
from unittest import TestCase

import numpy as np

from base import interpolate


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
