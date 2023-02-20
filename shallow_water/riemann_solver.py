from typing import Optional, Tuple

import defaults
import numpy as np

import core

from .core import Flux
from .wave_speed import MaximumWaveSpeed


class RiemannSolver(core.RiemannSolver):
    flux: core.FLUX
    gravitational_acceleration: float

    _scalar_wave_speed: core.WAVE_SPEED

    def __init__(
        self,
        gravitational_acceleration: Optional[float] = None,
        wave_speed=None,
    ):
        self.gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )
        self.flux = Flux(self.gravitational_acceleration)
        self._scalar_wave_speed = wave_speed or MaximumWaveSpeed(
            self.gravitational_acceleration
        )

    def wave_speed(self, value_left: np.ndarray, value_right: np.ndarray) -> Tuple:
        wave_speed_left, wave_speed_right = self._scalar_wave_speed(
            value_left, value_right
        )

        if isinstance(wave_speed_left, float):
            return wave_speed_left, wave_speed_right
        else:
            return wave_speed_left[:, None], wave_speed_right[:, None]

    @property
    def wave_speed_left(self) -> np.ndarray:
        return self._wave_speed_left[:, 0]

    @property
    def wave_speed_right(self) -> np.ndarray:
        return self._wave_speed_right[:, 0]
