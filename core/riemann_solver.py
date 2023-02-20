from typing import Any, Callable, Tuple

FLUX = Callable
WAVE_SPEED = Callable


class RiemannSolver:
    flux: FLUX
    wave_speed: WAVE_SPEED

    flux_left: ...
    flux_right: ...
    _wave_speed_left: ...
    _wave_speed_right: ...

    def __init__(
        self,
        flux: FLUX,
        wave_speed: WAVE_SPEED,
    ):
        self.flux = flux
        self.wave_speed = wave_speed

    def solve(self, value_left, value_right) -> Tuple[Any, Any]:
        self.flux_left, self.flux_right = self.flux(value_left), self.flux(value_right)
        self._wave_speed_left, self._wave_speed_right = self.wave_speed(
            value_left, value_right
        )

        factor = 1 / (self._wave_speed_right - self._wave_speed_left)
        intermediate_state = factor * (
            self._wave_speed_right * value_right
            - self._wave_speed_left * value_left
            - (self.flux_right - self.flux_left)
        )

        return (
            self.flux_left - self._wave_speed_left * (value_left - intermediate_state),
            intermediate_state,
        )

    @property
    def wave_speed_left(self):
        return self._wave_speed_left

    @property
    def wave_speed_right(self):
        return self._wave_speed_right
