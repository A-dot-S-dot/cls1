from typing import Generic, Optional, Tuple, TypeVar

import numpy as np
from base.decorator import HistoryDecorator

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
        self._time_step_history = np.array([])
        self._value_history = np.array([self.value])

    @property
    def time_history(self) -> np.ndarray:
        return self._time_history

    @property
    def time_step_history(self) -> np.ndarray:
        return self._time_step_history

    @property
    def value_history(self) -> np.ndarray:
        return self._value_history

    def update(self, time_step: float, solution: np.ndarray):
        DiscreteSolution.update(self, time_step, solution)
        self._time_history = np.append(self.time_history, self.time)
        self._time_step_history = np.append(self.time_step_history, time_step)
        self._value_history = np.append(
            self.value_history, np.array([solution.copy()]), axis=0
        )


# class CoarseSolution(DiscreteSolution):
#     """Coarsens a solution by taking it's solution and averaging regarding a
#     coarsened grid. To be more precise consider a discrete solution (ui). Then,
#     the coarsened solution is

#         Ui = 1/N*sum(uj, cell_j is in coarsened cell i)

#     where N denotes the number of fine cells which are in a coarsened cell. We
#     denote N COARSENING DEGREE. The AVERAGING_AXIS which axis should be averaged.

#     """

#     coarsening_degree: int

#     def __init__(self, solution: DiscreteSolution, coarsening_degree: int):
#         self.coarsening_degree = coarsening_degree
#         self._assert_admissible_coarsening_degree(solution)
#         self.time = solution.time.copy()
#         self.time_steps = solution.time_steps
#         self.values = self._coarsen_solution(solution.grid)
#         self.grid = self._coarsen_grid(solution.values)
#         self.space = self._coarsen_space(solution.space)

#     def _assert_admissible_coarsening_degree(self, solution: DiscreteSolution):
#         dof_length = solution.values.shape[1]
#         if dof_length % self.coarsening_degree != 0:
#             raise ValueError(
#                 f"Mesh size of {dof_length} is not divisible with respect to coarsening degree {self.coarsening_degree}."
#             )

#     def _coarsen_grid(self, grid: np.ndarray):
#         if len(grid) > 0:
#             coarse_grid = grid[0 :: self.coarsening_degree].copy()

#             for i in range(1, self.coarsening_degree):
#                 coarse_grid += grid[i :: self.coarsening_degree]

#             return 1 / self.coarsening_degree * coarse_grid
#         else:
#             return np.empty(0)

#     def _coarsen_solution(self, values: np.ndarray) -> np.ndarray:
#         coarse_values = values[:, 0 :: self.coarsening_degree].copy()

#         for i in range(1, self.coarsening_degree):
#             coarse_values += values[:, i :: self.coarsening_degree]

#         return 1 / self.coarsening_degree * coarse_values

#     def _coarsen_space(self, solver_space: SolverSpace) -> SolverSpace:
#         raise NotImplementedError
