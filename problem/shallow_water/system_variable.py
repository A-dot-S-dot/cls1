"""Contains problem dependent quantities."""
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from base.discretization.finite_volume import FiniteVolumeSpace


class NegativeHeightError(Exception):
    ...


class Nullifier:
    """Nullifies height and discharges below a certain threshold."""

    _epsilon: float

    def __init__(self, epsilon=1e-12):
        self._epsilon = epsilon

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return dof_vector * (dof_vector[:, 0] > self._epsilon)[:, None]


class DischargeToVelocityTransformer:
    """Returns for a given dof vector which contains heights and discharges a
    dof vector with heights and velocities. To be precise, we obtain (h, q/h).

    """

    _epsilon: float

    def __init__(self, epsilon=1e-12):
        self._epsilon = epsilon

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        height = dof_vector[:, 0].copy()
        discharge = dof_vector[:, 1].copy()

        if (height < 0).any():
            raise NegativeHeightError

        height[height < self._epsilon] = np.nan

        transformed_dof_vector = np.array([height, discharge / height]).T
        transformed_dof_vector[np.isnan(height)] = 0.0

        return transformed_dof_vector


class Flux:
    """Returns shallow water flux:

    (q, q**2/h+g*h**2/2)

    """

    _gravitational_acceleration: float
    _epsilon: float

    def __init__(self, gravitational_acceleration: float, epsilon=1e-12):
        self._gravitational_acceleration = gravitational_acceleration
        self._epsilon = epsilon

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        height = dof_vector[:, 0].copy()
        discharge = dof_vector[:, 1].copy()

        if (height < 0).any():
            raise NegativeHeightError

        height[height < self._epsilon] = np.nan

        flux = np.array(
            [
                discharge,
                discharge**2 / height
                + self._gravitational_acceleration * height**2 / 2,
            ]
        ).T
        flux[np.isnan(height)] = 0.0

        return flux


class WaveSpeed:
    """Contains local wave velocities for each  Riemann Problem, i.e. it contains

    left_wave = max(0, uL-sqrt(g*hL), uR-sqrt(g*hR))
    right_wave = max(0, uL+sqrt(g*hL), uR-+qrt(g*hR))

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

        return self._build_left_wave_velocities(
            height_left, height_right, velocity_left, velocity_right
        ), self._build_right_wave_velocities(
            height_left, height_right, velocity_left, velocity_right
        )

    def _build_left_wave_velocities(
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

    def _build_right_wave_velocities(
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


class SourceTermDiscretization(ABC):
    @abstractmethod
    def __call__(self, height_left, height_right, topography_step, step_length) -> Any:
        ...


class VanishingSourceTerm(SourceTermDiscretization):
    def __call__(self, height_left, height_right, topography_step, step_length) -> Any:
        return 0


class NaturalSouceTerm(SourceTermDiscretization):
    """Discretization of h*Db, i.e.

    h*Db=(hL+hR)/(2*step_length)*topography_step.

    """

    def __call__(self, height_left, height_right, topography_step, step_length) -> Any:
        return (height_left + height_right) / (2 * step_length) * topography_step
