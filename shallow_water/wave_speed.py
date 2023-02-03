from typing import Tuple

import numpy as np

from .core import DischargeToVelocityTransformer


class WaveSpeed:
    """Calculates local wave velocities for each  Riemann Problem, i.e. it contains

    wave_speed_left = max(0, uL-sqrt(g*hL), uR-sqrt(g*hR))
    wave_speed_right = max(0, uL+sqrt(g*hL), uR+sqrt(g*hR))

    """

    _gravitational_acceleration: float
    _discharge_to_velocity_transformer: DischargeToVelocityTransformer

    def __init__(self, gravitational_acceleration: float):
        self._gravitational_acceleration = gravitational_acceleration
        self._discharge_to_velocity_transformer = DischargeToVelocityTransformer()

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        height_velocity_left = self._discharge_to_velocity_transformer(value_left)
        height_velocity_right = self._discharge_to_velocity_transformer(value_right)

        height_left = height_velocity_left[:, 0]
        height_right = height_velocity_right[:, 0]
        velocity_left = height_velocity_left[:, 1]
        velocity_right = height_velocity_right[:, 1]

        return self._build_wave_speed_left(
            height_left, height_right, velocity_left, velocity_right
        ), self._build_wave_speed_right(
            height_left, height_right, velocity_left, velocity_right
        )

    def _build_wave_speed_left(
        self,
        height_left: np.ndarray,
        height_right: np.ndarray,
        velocity_left: np.ndarray,
        velocity_right: np.ndarray,
    ) -> np.ndarray:
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
        self,
        height_left: np.ndarray,
        height_right: np.ndarray,
        velocity_left: np.ndarray,
        velocity_right: np.ndarray,
    ) -> np.ndarray:
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
    _discharge_to_velocity_transformer: DischargeToVelocityTransformer

    def __init__(self, gravitational_acceleration: float):
        self._gravitational_acceleration = gravitational_acceleration
        self._discharge_to_velocity_transformer = DischargeToVelocityTransformer()

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        height_velocity_left = self._discharge_to_velocity_transformer(value_left)
        height_velocity_right = self._discharge_to_velocity_transformer(value_right)

        height_left = height_velocity_left[:, 0]
        height_right = height_velocity_right[:, 0]
        velocity_left = height_velocity_left[:, 1]
        velocity_right = height_velocity_right[:, 1]

        wave_speed = self._build_wave_speed(
            height_left, height_right, velocity_left, velocity_right
        )

        return -wave_speed, wave_speed

    def _build_wave_speed(
        self,
        height_left: np.ndarray,
        height_right: np.ndarray,
        velocity_left: np.ndarray,
        velocity_right: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            np.abs(velocity_left)
            + np.sqrt(self._gravitational_acceleration * height_left),
            np.abs(velocity_right)
            + np.sqrt(self._gravitational_acceleration * height_right),
        )
