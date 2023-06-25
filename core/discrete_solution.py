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
    def time(self) -> float:
        return self._time

    @property
    def value(self) -> np.ndarray:
        return self._value.copy()

    @property
    def dimension(self) -> float | Tuple[float, ...]:
        dimension = self.value.shape
        if len(dimension) == 1:
            return dimension[0]
        else:
            return dimension

    @property
    def space(self) -> Optional[T]:
        return self._space

    @property
    def grid(self) -> np.ndarray:
        return self.space.grid

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
    _solution: DiscreteSolution
    _time_history: np.ndarray
    _time_step_history: np.ndarray
    _value_history: np.ndarray

    def __init__(self, solution: DiscreteSolution):
        self._solution = solution
        self._time_history = np.array([self.time])
        self._value_history = np.array([self.value])

    @property
    def time(self) -> float:
        return self._solution.time

    @property
    def value(self) -> np.ndarray:
        return self._solution.value

    @property
    def space(self) -> T:
        return self._solution.space

    @property
    def time_history(self) -> np.ndarray:
        return self._time_history

    @property
    def time_step_history(self) -> np.ndarray:
        return np.diff(self.time_history)

    @property
    def value_history(self) -> np.ndarray:
        return self._value_history.copy()

    def update(self, time_step: float, value: np.ndarray):
        self._solution.update(time_step, value)
        self._time_history = np.append(self.time_history, self.time)
        self._value_history = np.append(
            self.value_history, np.array([value.copy()]), axis=0
        )

    def set_value(self, value: np.ndarray, time=0.0):
        values_past = self.time_history < time
        self._time_history = self.time_history[values_past]
        self._value_history = self.value_history[values_past]

        self.update(time - self.time, value)


class CoarseSolution(DiscreteSolution[T]):
    _coarsener: VectorCoarsener

    def __init__(self, solution: DiscreteSolution, coarsening_degree: int):
        self._solution = solution
        self._coarsener = VectorCoarsener(coarsening_degree)

    @property
    def time(self) -> float:
        return self._solution.time

    @property
    def value(self) -> np.ndarray:
        return self._coarsener(self._solution.value)

    @property
    def space(self) -> T:
        return self._solution.space.coarsen(self._coarsener.coarsening_degree)

    def update(self, time_step: float, value: np.ndarray):
        self._solution.update(time_step, value)

    def set_value(self, value: np.ndarray, time=0.0):
        self._solution.set_value(value, time=time)


class CoarseSolutionWithHistory(DiscreteSolutionWithHistory[T]):
    def __init__(self, solution: DiscreteSolution, coarsening_degree: int):
        self._coarsener = VectorCoarsener(coarsening_degree)
        super().__init__(solution)

    @property
    def value(self) -> np.ndarray:
        return self._coarsener(self._solution.value)

    @property
    def space(self) -> T:
        return self._solution.space.coarsen(self._coarsener.coarsening_degree)

    def update(self, time_step: float, value: np.ndarray):
        self._solution.update(time_step, value)
        self._time_history = np.append(self.time_history, self.time)
        self._value_history = np.append(
            self.value_history, np.array([self._coarsener(value)]), axis=0
        )
