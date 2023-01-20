from typing import Generic, Optional, Tuple, TypeVar

import numpy as np

from .abstract import SolverSpace

T = TypeVar("T", bound=SolverSpace)


class DiscreteSolution(Generic[T]):
    """Class representing time dependent discrete solution."""

    _time: float
    _value: np.ndarray

    _grid: Optional[np.ndarray]
    _space: Optional[T]

    def __init__(
        self,
        initial_value: np.ndarray,
        start_time=0.0,
        grid=None,
        solver_space=None,
    ):
        self._time = start_time
        self._value = initial_value
        self._grid = grid
        self._space = solver_space

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
    def grid(self) -> np.ndarray:
        if self._grid is not None:
            return self._grid
        else:
            raise AttributeError("Grid attribute does not exist.")

    @property
    def space(self) -> T:
        if self._space is not None:
            return self._space
        else:
            raise AttributeError("Solver space attribute does not exist.")

    @property
    def value(self) -> np.ndarray:
        return self._value.copy()

    def update(self, time_step: float, solution: np.ndarray):
        self._time += time_step
        self._value = solution

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(time={self.time}, value={self.value}, grid={self.grid}, space={self.space})"
        )


class DiscreteSolutionWithHistory(DiscreteSolution[T]):
    _time_history: np.ndarray
    _time_step_history: np.ndarray
    _value_history: np.ndarray

    def __init__(
        self,
        initial_value: np.ndarray,
        start_time=0.0,
        grid=None,
        solver_space=None,
    ):
        DiscreteSolution.__init__(
            self,
            initial_value,
            start_time=start_time,
            grid=grid,
            solver_space=solver_space,
        )

        self._time_history = np.array([self.time])
        self._value_history = np.array([self.value])

    @property
    def time_history(self) -> np.ndarray:
        return self._time_history

    @property
    def time_step_history(self) -> np.ndarray:
        return self.time_history[1:] - self.time_history[:-1]

    @property
    def value_history(self) -> np.ndarray:
        return self._value_history

    def update(self, time_step: float, solution: np.ndarray):
        DiscreteSolution.update(self, time_step, solution)
        self._time_history = np.append(self.time_history, self.time)
        self._value_history = np.append(
            self.value_history, np.array([solution.copy()]), axis=0
        )
