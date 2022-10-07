from unittest import TestCase

from pde_solver.interpolate import CellAverageInterpolator
from test.test_helper import LINEAR_MESH


class TestInterpolator(TestCase):
    def test_interpolator(self):
        f = lambda _: 1
        interpolator = CellAverageInterpolator(LINEAR_MESH, 1)
        interpolation = interpolator.interpolate(f)
        expected_interpolation = [1, 1, 1, 1]

        self.assertListEqual(list(interpolation), expected_interpolation)
