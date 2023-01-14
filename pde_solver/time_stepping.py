from typing import Callable
import numpy as np


class TimeStepTooSmallError(Exception):
    ...


class TimeStepGenerator:
    _time_step: float = np.nan
    _cfl_number: float
    _adaptive: bool
    _get_time_step: Callable[[], float]

    def __init__(
        self,
        time_step_function: Callable[[], float],
        cfl_number: float,
        adaptive=False,
    ):
        self._adaptive = adaptive
        self._cfl_number = cfl_number

        if adaptive:
            self._get_time_step = lambda: time_step_function() * self._cfl_number
        else:
            self._time_step = time_step_function()
            self._get_time_step = lambda: self._time_step * self._cfl_number

    def __call__(self) -> float:
        return self._get_time_step()

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(time_step={self._time_step:.1e}, cfl={self._cfl_number}, adaptive={self._adaptive})"
        )


class TimeStepping:
    """Iterator for time stepping used for solving PDEs."""

    time: float
    time_steps: int

    _start_time: float
    _end_time: float
    _time_step_generator: TimeStepGenerator
    _time_step_function: Callable[[], float]
    _length: int = 0
    _current_time_step: float = 0
    _stop_iteration: bool

    def __init__(
        self,
        end_time: float,
        cfl_number: float,
        time_step_function: Callable[[], float],
        adaptive=False,
        start_time=0,
    ):
        self._start_time = start_time
        self._end_time = end_time
        self._time_step_function = time_step_function
        self._time_step_generator = TimeStepGenerator(
            time_step_function, cfl_number, adaptive
        )

        if not adaptive:
            self._build_length()

    def _build_length(self):
        time_step = self._time_step_generator()
        duration = self._end_time - self._start_time

        self._length = int(duration // time_step)
        self._length += 1 if time_step * self._length < duration - 1e-12 else 0

    def __iter__(self):
        self._stop_iteration = False
        self.time = self._start_time
        self.time_steps = 0
        return self

    def __next__(self) -> float:
        if self._stop_iteration:
            raise StopIteration
        else:
            return self._get_time_step()

    def __len__(self) -> int:
        return self._length

    def _get_time_step(self) -> float:
        time_step = min(self._time_step_generator(), self._end_time - self.time)
        self._update_time(time_step)

        return time_step

    def _update_time(self, time_step: float):
        if time_step < 1e-12:
            raise TimeStepTooSmallError(f"time step {time_step} is too small.")

        self.time += time_step
        self.time_steps += 1
        self._current_time_step = time_step

        if self._end_time - self.time < 1e-12:
            self.time = self._end_time
            self._stop_iteration = True

    def satisfy_cfl_condition(self) -> bool:
        return self._current_time_step <= self._time_step_function()

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(start_time={self._start_time}, end_time={self._end_time}, time_step_generator={self._time_step_generator})"
        )
