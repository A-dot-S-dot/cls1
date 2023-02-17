import core
import numpy as np


class LocalMaximum:
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    """

    _neighbour_indices: np.ndarray

    def __init__(self, neighbour_indices: core.NeighbourIndicesMapping):
        self._neighbour_indices = neighbour_indices.array

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return np.amax(
            dof_vector[self._neighbour_indices],
            axis=1,
        )


class LocalMinimum(LocalMaximum):
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    """

    _neighbour_indices: np.ndarray

    def __init__(self, neighbour_indices: core.NeighbourIndicesMapping):
        self._neighbour_indices = neighbour_indices.array

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return np.amin(
            dof_vector[self._neighbour_indices],
            axis=1,
        )
