from unittest import TestCase

import numpy as np

import shallow_water
from numpy.testing import assert_almost_equal


class TestWaveSpeed(TestCase):
    wave_speed = shallow_water.WaveSpeed(1)

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
