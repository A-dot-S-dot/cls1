from unittest import TestCase

import numpy as np
import shallow_water
from numpy.testing import assert_almost_equal


class TestRiemannSolver(TestCase):
    def test_system_input(self):
        solver = shallow_water.RiemannSolver(1.0)
        value_left = np.array(
            [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
        )
        value_right = np.array(
            [[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]
        )

        flux, bar_state = solver.solve(value_left, value_right)
        expected_flux = np.array(
            [[1.5, 0.75], [1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75], [1.5, 0.75]]
        )
        expected_bar_state = np.array(
            [
                [1.25, 0.625],
                [0.75, 0.875],
                [0.75, -0.875],
                [1.25, -0.625],
                [1.25, 0.625],
            ]
        )

        assert_almost_equal(flux, expected_flux)
        assert_almost_equal(bar_state, expected_bar_state)

    def test_scalar_input(self):
        solver = shallow_water.RiemannSolver(1.0)
        value_left = np.array([2.0, 0.0])
        value_right = np.array([1.0, 1.0])
        flux, bar_state = solver.solve(value_left, value_right)
        expected_flux = np.array([1.5, 0.75])
        expected_bar_state = np.array([1.25, 0.625])

        assert_almost_equal(flux, expected_flux)
        assert_almost_equal(bar_state, expected_bar_state)
