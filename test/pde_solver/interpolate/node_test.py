from unittest import TestCase

from pde_solver.interpolate import NodeValuesInterpolator
from test.test_helper import LINEAR_LAGRANGE_SPACE


class TestInterpolator(TestCase):
    def test_interpolator(self):
        f = lambda _: 1
        interpolator = NodeValuesInterpolator(*LINEAR_LAGRANGE_SPACE.basis_nodes)
        interpolation = interpolator.interpolate(f)
        expected_interpolation = [1, 1, 1, 1]

        self.assertListEqual(list(interpolation), expected_interpolation)
