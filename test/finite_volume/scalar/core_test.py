from unittest import TestCase

import numpy as np
from finite_volume.scalar.core import *
from numpy.testing import assert_equal


class TestWaveSpeed(TestCase):
    def test_advection(self):
        wave_speed = get_wave_speed("advection")
        value_left = np.array([1.0, 2.0, 3.0])
        value_right = np.array([2.0, 3.0, 4.0])

        expected_wave_speed = np.array([1.0, 1.0, 1.0])

        wl, wr = wave_speed(value_left, value_right)

        assert_equal(wl.shape, (3,))
        assert_equal(wr.shape, (3,))
        assert_equal(wl, -expected_wave_speed)
        assert_equal(wr, expected_wave_speed)
