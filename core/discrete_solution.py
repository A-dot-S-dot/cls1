from typing import Generic, Optional, Tuple, TypeVar

import numpy as np

from .space import SolverSpace
from .vector_coarsener import VectorCoarsener

T = TypeVar("T", bound=SolverSpace)


class DiscreteSolution(Generic[T]):
    """Class representing time dependent discrete solution."""

    _time: float
    _value: np.ndarray

    _space: Optional[T]

    def __init__(
        self,
        initial_value: np.ndarray,
        initial_time=0.0,
        space=None,
    ):
        self._time = initial_time
        self._value = initial_value
        self._space = space

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
        if self._space is not None:
            return self._space.grid
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

    def update(self, time_step: float, value: np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("Solution is not finite.")

        self._time += time_step
        self._value = value

    def set_value(self, value: np.ndarray, time=0.0):
        self.update(time - self._time, value)

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
        initial_time=0.0,
        space=None,
    ):
        DiscreteSolution.__init__(
            self,
            initial_value,
            initial_time=initial_time,
            space=space,
        )

        self._time_history = np.array([self.time])
        self._value_history = np.array([self.value])

    @property
    def time_history(self) -> np.ndarray:
        return self._time_history.copy()

    @property
    def time_step_history(self) -> np.ndarray:
        return np.diff(self.time_history)

    @property
    def value_history(self) -> np.ndarray:
        return self._value_history.copy()

    def update(self, time_step: float, solution: np.ndarray):
        DiscreteSolution.update(self, time_step, solution)
        self._time_history = np.append(self.time_history, self.time)
        self._value_history = np.append(
            self.value_history, np.array([solution.copy()]), axis=0
        )

    def set_value(self, value: np.ndarray, time=0.0):
        values_past = self._time_history < time
        self._time_history = self._time_history[values_past]
        self._value_history = self._value_history[values_past]

        self.update(time - self._time, value)


class CoarseSolution(DiscreteSolution):
    _coarsener: VectorCoarsener

    def __init__(self, solution: DiscreteSolution, coarsening_degree: int, space=None):
        self._time = solution.time

        self._coarsener = VectorCoarsener(coarsening_degree)
        self._value = self._coarsener(solution.value)

        self._space = space


class CoarseSolutionWithHistory(DiscreteSolutionWithHistory, CoarseSolution):
    def __init__(
        self, solution: DiscreteSolutionWithHistory, coarsening_degree: int, space=None
    ):
        CoarseSolution.__init__(self, solution, coarsening_degree, space=space)
        self._time_history = solution.time_history
        self._value_history = np.array(
            [self._coarsener(dof) for dof in solution.value_history]
        )
