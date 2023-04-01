import numpy as np


class VectorCoarsener:
    """Coarsens a vector by taking it's values and averaging regarding a
    coarsened grid. To be more precise consider a vector (ui). Then, the
    coarsened vector is defined by

        Ui = 1/N*sum(uj, cell_j is in coarsened cell i)

    where N denotes the number of fine cells which are in a coarsened cell. We
    denote N COARSENING DEGREE.

    """

    coarsening_degree: int

    def __init__(self, coarsening_degree: int):
        self.coarsening_degree = coarsening_degree

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self._assert_admissible_vector(vector)

        return self._coarsen_vector(vector)

    def _assert_admissible_vector(self, vector: np.ndarray):
        if len(vector) % self.coarsening_degree != 0:
            raise ValueError(
                f"Vector with {len(vector)} entries is not divisible with respect to oarsening degree {self.coarsening_degree}."
            )

    def _coarsen_vector(self, vector: np.ndarray) -> np.ndarray:
        coarse_values = vector[0 :: self.coarsening_degree].copy()

        for i in range(1, self.coarsening_degree):
            coarse_values += vector[i :: self.coarsening_degree]

        return 1 / self.coarsening_degree * coarse_values
