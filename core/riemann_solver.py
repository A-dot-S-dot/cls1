from typing import Callable, Tuple

import numpy as np

FLUX = Callable[[np.ndarray], np.ndarray]
WAVE_SPEED = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class RiemannSolver:
    flux: FLUX
    wave_speed: WAVE_SPEED

    flux_left: np.ndarray
    flux_right: np.ndarray
    _wave_speed_left: np.ndarray
    _wave_speed_right: np.ndarray

    def __init__(
        self,
        flux: FLUX,
        wave_speed: WAVE_SPEED,
    ):
        self.flux = flux
        self.wave_speed = wave_speed

    def solve(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.flux_left, self.flux_right = self.flux(value_left), self.flux(value_right)
        self._wave_speed_left, self._wave_speed_right = self.wave_speed(
            value_left, value_right
        )

        factor = 1 / (self._wave_speed_right + -self._wave_speed_left)
        intermediate_state = factor * (
            self._wave_speed_right * value_right
            + -self._wave_speed_left * value_left
            - (self.flux_right + -self.flux_left)
        )

        return intermediate_state, self.flux_left - self._wave_speed_left * (
            value_left - intermediate_state
        )

    @property
    def wave_speed_left(self) -> np.ndarray:
        return self._wave_speed_left

    @property
    def wave_speed_right(self) -> np.ndarray:
        return self._wave_speed_right
