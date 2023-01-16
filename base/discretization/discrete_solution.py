from typing import List, Optional, Tuple

import numpy as np

from .abstract import SolverSpace, EmptySpace


class DiscreteSolution:
    """Class representing time dependent discrete solution."""

    _time: float
    _value: np.ndarray

    _grid: Optional[np.ndarray]
    _space: Optional[SolverSpace]

    _save_history: bool
    _time_history: np.ndarray
    _time_step_history: np.ndarray
    _value_history: np.ndarray

    def __init__(
        self,
        initial_value: np.ndarray,
        start_time=0.0,
        grid=None,
        solver_space=None,
        save_history=False,
    ):
        self._time = start_time
        self._value = initial_value
        self._grid = grid
        self._space = solver_space
        self._save_history = save_history

        if save_history:
            self._build_history_attributes()

    def _build_history_attributes(self):
        self._time_history = np.array([self.time])
        self._time_step_history = np.array([])
        self._value_history = np.array([self.value])

    @property
    def dimension(self) -> float | Tuple[float, ...]:
        dimension = self.value.shape
        if len(dimension) == 1:
            return dimension[0]
        else:
            return dimension

    @property
    def time(self) -> float:
        return self._time

    @property
    def time_history(self) -> np.ndarray:
        return self._time_history

    @property
    def time_step_history(self) -> np.ndarray:
        return self._time_step_history

    @property
    def grid(self) -> np.ndarray:
        if self._grid is not None:
            return self._grid
        else:
            raise AttributeError("Grid attribute does not exist.")

    @property
    def space(self) -> SolverSpace:
        if self._space is not None:
            return self._space
        else:
            raise AttributeError("Solver space attribute does not exist.")

    @property
    def value(self) -> np.ndarray:
        return self._value.copy()

    @property
    def value_history(self) -> np.ndarray:
        return self._value_history

    def update(self, time_step: float, solution: np.ndarray):
        self._time += time_step
        self._value = solution

        if self._save_history:
            self._time_history = np.append(self.time_history, self.time)
            self._time_step_history = np.append(self.time_step_history, time_step)
            self._value_history = np.append(
                self.value_history, np.array([self.value]), axis=0
            )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(grid={self.grid}, time={self.time_history}, values={self.value}, time_steps={self.time_step_history})"
        )
