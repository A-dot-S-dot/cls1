from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from base.discretization.discrete_solution import DiscreteSolution


class TimeStepTooSmallError(Exception):
    ...


class CFLConditionViolatedError(Exception):
    ...


class DiscreteSolutionDependentTimeStep:
    _time_step: Callable[[np.ndarray], float]
    _discrete_solution: DiscreteSolution

    def __init__(
        self,
        time_step: Callable[[np.ndarray], float],
        discrete_solution: DiscreteSolution,
    ):
        self._time_step = time_step
        self._discrete_solution = discrete_solution

    def __call__(self) -> float:
        return self._time_step(self._discrete_solution.value)


class CFLChecker:
    _optimal_time_step: Callable[[np.ndarray], float]

    def __init__(self, optimal_time_step: Callable[[np.ndarray], float]):
        self._optimal_time_step = optimal_time_step

    def __call__(self, time_step, *dof_vector: np.ndarray):
        for dofs in dof_vector:
            optimal_time_step = self._optimal_time_step(dofs)
            if time_step > optimal_time_step:
                raise CFLConditionViolatedError


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

    @property
    def adaptive(self) -> bool:
        return self._adaptive

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
    time_steps_number: int

    _start_time: float
    _end_time: float
    _constant_time_step: int = 0
    _stop_iteration: bool

    def __init__(
        self,
        end_time: float,
        cfl_number: float,
        optimal_time_step: Callable[[], float],
        adaptive=False,
        start_time=0.0,
    ):
        self._start_time = start_time
        self._end_time = end_time
        self._build_optimal_time_step = optimal_time_step
        self._time_step_generator = TimeStepGenerator(
            optimal_time_step, cfl_number, adaptive
        )

    def __iter__(self):
        self._stop_iteration = False
        self.time = self._start_time
        self.time_steps_number = 0
        return self

    def __next__(self) -> float:
        if self._stop_iteration:
            raise StopIteration
        else:
            return self._build_time_step()

    def __len__(self) -> int:
        if self._time_step_generator.adaptive:
            return 0
        else:
            time_step = self._time_step_generator()
            duration = self._end_time - self._start_time

            time_steps_number = int(duration // time_step)
            time_steps_number += (
                1 if time_step * time_steps_number < duration - 1e-12 else 0
            )
            return time_steps_number

    def _build_time_step(self) -> float:
        time_step = min(self._time_step_generator(), self._end_time - self.time)

        self._check_too_small_time_step(time_step)
        self._update_time(time_step)
        self._check_end_reached()

        return time_step

    def _check_too_small_time_step(self, time_step: float):
        if time_step < 1e-12:
            raise TimeStepTooSmallError(f"time step {time_step} is too small.")

    def _update_time(self, time_step: float):
        self.time += time_step
        self.time_steps_number += 1

    def _check_end_reached(self):
        if self._end_time - self.time < 1e-12:
            self.time = self._end_time
            self._stop_iteration = True

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(start_time={self._start_time}, end_time={self._end_time}, time_step_generator={self._time_step_generator})"
        )
