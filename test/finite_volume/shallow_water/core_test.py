from unittest import TestCase

import numpy as np
from finite_volume.shallow_water.core import *
from numpy.testing import assert_almost_equal, assert_equal


class TestAverage(TestCase):
    def test_scalar_average(self):
        value_left = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        value_right = np.array([1.0, 0.0, 1.0, 2.0, 1.0])
        expected_output = [1.5, 0.5, 0.5, 1.5, 1.5]

        assert_equal(get_average(value_left, value_right), expected_output)

    def test_system_average(self):
        value_left = np.array(
            [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
        )
        value_right = np.array(
            [[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]
        )
        expected_output = [[1.5, 0.5], [0.5, 0.5], [0.5, -0.5], [1.5, -0.5], [1.5, 0.5]]

        assert_equal(get_average(value_left, value_right), expected_output)


class TestNullify(TestCase):
    def test_nullify(self):
        dof_vector = np.array([[0.5, 2.0], [0.5, 2.0], [2.0, 0.5], [2.0, 2.0]])
        expected_output = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

        assert_equal(nullify(dof_vector, eps=1.0), expected_output)


class TestGetVelocity(TestCase):
    def test_transformer(self):
        dof_vector = np.array([[2.0, -4.0], [2.0, 4.0], [0.0, 0.0]])
        expected_output = np.array([-2.0, 2.0, 0.0])

        assert_almost_equal(get_velocity(dof_vector), expected_output)


class TestIsConstant(TestCase):
    def test_not_constant(self):
        bottom = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertFalse(is_constant(bottom))

    def test_is_constant(self):
        bottom = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(is_constant(bottom))


class TestGetHeightPositivityFix(TestCase):
    def test_fix(self):
        bar_state = np.array([1.0, 1.75, 2.0])
        bathymetry_step = np.array([0.0, -1.0, 0.0])
        expected_fix = np.array([0.0, -0.5, 0.0])

        assert_equal(
            get_height_positivity_fix(bar_state, bathymetry_step), expected_fix
        )


class TestFlux(TestCase):
    def test_flux(self):
        flux = Flux(1)
        dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
        expected_flux = np.array([[1.0, 1.5], [0.0, 0.0], [-1.0, 1.5], [0.0, 2.0]])

        assert_equal(flux(dof_vector), expected_flux)

    def test_negative_height_error(self):
        self.assertRaises(ValueError, Flux(1), np.array([[-1, 2], [2, 3]]))
        self.assertRaises(ValueError, Flux(1), np.array([-1, 2]))

    def test_scalar_flux(self):
        flux = Flux(1)
        input = np.array([1.0, 1.0])
        expected_flux = np.array([1.0, 1.5])

        assert_equal(flux(input), expected_flux)


class TestWaveSpeed(TestCase):
    wave_speed = WaveSpeed(1)

    def test_wave_speed(self):
        value_left = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
        value_right = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
        wave_speed_left, wave_speed_right = self.wave_speed(value_left, value_right)
        expected_wave_speed_left = np.array([-1.41421356, 0.0, -2.0, -2.0])
        expected_wave_speed_right = np.array([2.0, 2.0, 0.0, 1.41421356])

        assert_almost_equal(wave_speed_left, expected_wave_speed_left)
        assert_almost_equal(wave_speed_right, expected_wave_speed_right)

    def test_scalar_wave_speed(self):
        value_left = np.array([2.0, 0.0])
        value_right = np.array([1.0, 1.0])

        wave_speed_left, wave_speed_right = self.wave_speed(value_left, value_right)
        expected_wave_speed_left = -1.41421356
        expected_wave_speed_right = 2.0

        self.assertAlmostEqual(wave_speed_left, expected_wave_speed_left)
        self.assertAlmostEqual(wave_speed_right, expected_wave_speed_right)


class TestRiemannSolver(TestCase):
    def test_system_input(self):
        solver = RiemannSolver(1.0)
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
        solver = RiemannSolver(1.0)
        value_left = np.array([2.0, 0.0])
        value_right = np.array([1.0, 1.0])
        flux, bar_state = solver.solve(value_left, value_right)
        expected_flux = np.array([1.5, 0.75])
        expected_bar_state = np.array([1.25, 0.625])

        assert_almost_equal(flux, expected_flux)
        assert_almost_equal(bar_state, expected_bar_state)
