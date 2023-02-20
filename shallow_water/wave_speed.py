from typing import Tuple

import numpy as np

from .core import get_heights, get_velocities


class WaveSpeed:
    """Calculates local wave velocities for each  Riemann Problem, i.e. it contains

    wave_speed_left = max(0, uL-sqrt(g*hL), uR-sqrt(g*hR))
    wave_speed_right = max(0, uL+sqrt(g*hL), uR+sqrt(g*hR))

    """

    _gravitational_acceleration: float

    def __init__(self, gravitational_acceleration: float):
        self._gravitational_acceleration = gravitational_acceleration

    def __call__(self, value_left, value_right) -> Tuple:
        height_left, height_right = get_heights(value_left, value_right)
        velocity_left, velocity_right = get_velocities(value_left, value_right)

        return self._build_wave_speed_left(
            height_left, height_right, velocity_left, velocity_right
        ), self._build_wave_speed_right(
            height_left, height_right, velocity_left, velocity_right
        )

    def _build_wave_speed_left(
        self, height_left, height_right, velocity_left, velocity_right
    ):
        return np.minimum(
            np.minimum(
                velocity_left
                + -np.sqrt(self._gravitational_acceleration * height_left),
                velocity_right
                + -np.sqrt(self._gravitational_acceleration * height_right),
            ),
            0,
        )

    def _build_wave_speed_right(
        self, height_left, height_right, velocity_left, velocity_right
    ):
        return np.maximum(
            np.maximum(
                velocity_left + np.sqrt(self._gravitational_acceleration * height_left),
                velocity_right
                + np.sqrt(self._gravitational_acceleration * height_right),
            ),
            0,
        )


class MaximumWaveSpeed:
    """Calculates local wave velocities for each  Riemann Problem, i.e. it contains

    wave_speed = max(abs(uL)+sqrt(g*hL), |uR| + sqrt(g*hR))

    """

    _gravitational_acceleration: float

    def __init__(self, gravitational_acceleration: float):
        self._gravitational_acceleration = gravitational_acceleration

    def __call__(self, value_left, value_right) -> Tuple:
        height_left, height_right = get_heights(value_left, value_right)
        velocity_left, velocity_right = get_velocities(value_left, value_right)

        wave_speed = self._build_wave_speed(
            height_left, height_right, velocity_left, velocity_right
        )

        return -wave_speed, wave_speed

    def _build_wave_speed(
        self, height_left, height_right, velocity_left, velocity_right
    ):
        return np.maximum(
            np.abs(velocity_left)
            + np.sqrt(self._gravitational_acceleration * height_left),
            np.abs(velocity_right)
            + np.sqrt(self._gravitational_acceleration * height_right),
        )
