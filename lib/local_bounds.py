import numpy as np
from core import SolverSpace


class LocalMaximum:
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    """

    _space: SolverSpace

    def __init__(self, space: SolverSpace):
        self._space = space

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return np.amax(
            dof_vector[self._space.dof_neighbours.array],
            axis=1,
        )


class LocalMinimum(LocalMaximum):
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    """

    _space: SolverSpace

    def __init__(self, space: SolverSpace):
        self._space = space

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return np.amin(
            dof_vector[self._space.dof_neighbours.array],
            axis=1,
        )
