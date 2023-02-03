from typing import Optional, Tuple

import defaults
import numpy as np

import core
from core import finite_volume

from .core import Flux
from .wave_speed import MaximumWaveSpeed


class RiemannSolver(core.RiemannSolver):
    _flux: core.FLUX
    _scalar_wave_speed: core.WAVE_SPEED
    _conditions_applier: core.finite_volume.BoundaryConditionsApplier
    _gravitational_acceleration: float
    _gravitational_acceleration: float
    _scalar_wave_speed: core.WAVE_SPEED

    def __init__(
        self,
        conditions_applier=None,
        gravitational_acceleration: Optional[float] = None,
        wave_speed=None,
    ):
        self._conditions_applier = (
            conditions_applier or finite_volume.PeriodicBoundaryConditionsApplier()
        )
        assert self._conditions_applier.cells_to_add_numbers == (1, 1)

        self._gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )
        self._flux = Flux(self._gravitational_acceleration)
        self._scalar_wave_speed = wave_speed or MaximumWaveSpeed(
            self._gravitational_acceleration
        )

    def _wave_speed(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        wave_speed_left, wave_speed_right = self._scalar_wave_speed(
            value_left, value_right
        )

        return wave_speed_left[:, None], wave_speed_right[:, None]

    @property
    def gravitational_acceleration(self) -> float:
        return self._gravitational_acceleration

    @property
    def wave_speed_left(self) -> np.ndarray:
        return self._wave_speed_left[:, 0]

    @property
    def wave_speed_right(self) -> np.ndarray:
        return self._wave_speed_right[:, 0]
