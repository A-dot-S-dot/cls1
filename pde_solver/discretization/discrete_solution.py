from typing import List, Tuple
from itertools import product

import numpy as np


class DiscreteSolution:
    """Class representing discrete solution."""

    time: np.ndarray
    time_steps: List[float]
    grid: np.ndarray
    values: np.ndarray

    def __init__(self, start_time: float, initial_data: np.ndarray, grid: np.ndarray):
        """The first dimension should correspond to the number of time steps and
        the second to DOFs dimension.

        """
        self.time = np.array([start_time])
        self.time_steps = []
        self.grid = grid
        self.values = np.array([initial_data])

    @property
    def dimension(self) -> float | Tuple[float, ...]:
        dimension = self.initial_data.shape
        if len(dimension) == 1:
            return dimension[0]
        else:
            return dimension

    @property
    def initial_data(self) -> np.ndarray:
        return self.values[0].copy()

    @property
    def end_values(self) -> np.ndarray:
        return self.values[-1].copy()

    @property
    def start_time(self) -> float:
        return self.time[0]

    @property
    def end_time(self) -> float:
        return self.time[-1]

    def add_solution(self, time_step: float, solution: np.ndarray):
        new_time = self.time[-1] + time_step

        self.time = np.append(self.time, new_time)
        self.time_steps.append(time_step)
        self.values = np.append(self.values, np.array([solution]), axis=0)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(grid={self.grid}, time={self.time}, values={self.values}, time_steps={self.time_steps})"
        )


class TemporalInterpolation:
    """Interpolate discrete solution values for diffrent times."""

    def __call__(self, solution: DiscreteSolution, new_time: np.ndarray) -> np.ndarray:
        interpolated_values = np.empty((len(new_time), *solution.initial_data.shape))

        for index in product(*[range(dim) for dim in solution.initial_data.shape]):
            # array[(slice(start, end))]=array[:]
            interpolated_values[(slice(0, len(new_time)), *index)] = np.interp(
                new_time,
                solution.time,
                solution.values[(slice(0, len(solution.time)), *index)],
            )

        return interpolated_values


class CoarseSolution(DiscreteSolution):
    """Coarsens a solution by taking it's solution and averaging regarding a
    coarsened grid. To be more precise consider a discrete solution (ui). Then,
    the coarsened solution is

        Ui = 1/N*sum(uj, cell_j is in coarsened cell i)

    where N denotes the number of fine cells which are in a coarsened cell. We
    denote N COARSENING DEGREE. The AVERAGING_AXIS which axis should be averaged.

    """

    coarsening_degree: int

    def __init__(self, solution: DiscreteSolution, coarsening_degree: int):
        self.coarsening_degree = coarsening_degree
        self._assert_admissible_coarsening_degree(solution)
        self.time = solution.time.copy()
        self.time_steps = solution.time_steps
        self.grid = self._coarsen_grid(solution)
        self.values = self._coarsen_solution(solution)

    def _assert_admissible_coarsening_degree(self, solution: DiscreteSolution):
        dof_length = solution.values.shape[1]
        if dof_length % self.coarsening_degree != 0:
            raise ValueError(
                f"Mesh size of {dof_length} is not divisible with respect to coarsening degree {self.coarsening_degree}."
            )

    def _coarsen_grid(self, solution: DiscreteSolution):
        coarse_grid = solution.grid[0 :: self.coarsening_degree].copy()

        for i in range(1, self.coarsening_degree):
            coarse_grid += solution.grid[i :: self.coarsening_degree]

        return 1 / self.coarsening_degree * coarse_grid

    def _coarsen_solution(self, solution: DiscreteSolution) -> np.ndarray:
        coarse_values = solution.values[:, 0 :: self.coarsening_degree].copy()

        for i in range(1, self.coarsening_degree):
            coarse_values += solution.values[:, i :: self.coarsening_degree]

        return 1 / self.coarsening_degree * coarse_values
