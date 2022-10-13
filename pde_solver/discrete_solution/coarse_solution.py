import numpy as np

from .discrete_solution import DiscreteSolution


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
        self.time = solution.time
        self.coarsening_degree = coarsening_degree
        self._assert_admissible_coarsening_degree(solution)
        self.values = self._coarsen_solution(solution)

    def _assert_admissible_coarsening_degree(self, solution: DiscreteSolution):
        dof_length = solution.values.shape[1]
        if dof_length % self.coarsening_degree != 0:
            raise ValueError(
                f"Mesh size of {dof_length} is not divisible with respect to coarsening degree {self.coarsening_degree}."
            )

    def _coarsen_solution(self, discrete_solution: DiscreteSolution) -> np.ndarray:
        solution_values = discrete_solution.values
        coarse_values = solution_values[:, 0 :: self.coarsening_degree]

        for i in range(1, self.coarsening_degree):
            coarse_values += solution_values[:, i :: self.coarsening_degree]

        return 1 / self.coarsening_degree * coarse_values
