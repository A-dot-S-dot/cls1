from typing import Tuple

import numpy as np

from core import FiniteVolumeSpace

from .core import DischargeToVelocityTransformer


class WaveSpeed:
    """Calculates local wave velocities for each  Riemann Problem, i.e. it contains

    wave_speed_left = max(0, uL-sqrt(g*hL), uR-sqrt(g*hR))
    wave_speed_right = max(0, uL+sqrt(g*hL), uR+sqrt(g*hR))

    """

    _volume_space: FiniteVolumeSpace
    _gravitational_acceleration: float
    _discharge_to_velocity_transformer: DischargeToVelocityTransformer

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        discharge_to_velocity_transformer=None,
    ):
        self._volume_space = volume_space
        self._gravitational_acceleration = gravitational_acceleration
        self._discharge_to_velocity_transformer = (
            discharge_to_velocity_transformer or DischargeToVelocityTransformer()
        )

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height_velocity_dof_vector = self._discharge_to_velocity_transformer(dof_vector)

        height_left = height_velocity_dof_vector[
            self._volume_space.left_cell_indices, 0
        ]
        height_right = height_velocity_dof_vector[
            self._volume_space.right_cell_indices, 0
        ]
        velocity_left = height_velocity_dof_vector[
            self._volume_space.left_cell_indices, 1
        ]
        velocity_right = height_velocity_dof_vector[
            self._volume_space.right_cell_indices, 1
        ]

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

    _volume_space: FiniteVolumeSpace
    _gravitational_acceleration: float
    _discharge_to_velocity_transformer: DischargeToVelocityTransformer

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        discharge_to_velocity_transformer=None,
    ):
        self._volume_space = volume_space
        self._gravitational_acceleration = gravitational_acceleration
        self._discharge_to_velocity_transformer = (
            discharge_to_velocity_transformer or DischargeToVelocityTransformer()
        )

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        height_velocity_dof_vector = self._discharge_to_velocity_transformer(dof_vector)

        height_left = height_velocity_dof_vector[
            self._volume_space.left_cell_indices, 0
        ]
        height_right = height_velocity_dof_vector[
            self._volume_space.right_cell_indices, 0
        ]
        velocity_left = height_velocity_dof_vector[
            self._volume_space.left_cell_indices, 1
        ]
        velocity_right = height_velocity_dof_vector[
            self._volume_space.right_cell_indices, 1
        ]

        return self._build_wave_speed(
            height_left, height_right, velocity_left, velocity_right
        )

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
