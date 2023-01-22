from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np

import shallow_water


class TestWaveSpeed(TestCase):
    volume_space = VOLUME_SPACE
    wave_speed = shallow_water.WaveSpeed(volume_space, 1)

    def test_wave_speed(self):
        dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
        wave_speed_left, wave_speed_right = self.wave_speed(dof_vector)
        expected_wave_speed_left = np.array([-1.41421356, 0.0, -2.0, -2.0])
        expected_wave_speed_right = np.array([2.0, 2.0, 0.0, 1.41421356])

        for i in range(self.volume_space.node_number):
            self.assertAlmostEqual(
                wave_speed_left[i],
                expected_wave_speed_left[i],
                msg=f"left wave speed, index={i}",
            )
            self.assertAlmostEqual(
                wave_speed_right[i],
                expected_wave_speed_right[i],
                msg=f"right wave speed, index={i}",
            )
