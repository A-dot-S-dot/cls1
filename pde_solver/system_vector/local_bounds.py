from pde_solver.solver_space import LagrangeFiniteElementSpace
from .system_vector import SystemVector

import numpy as np


class LocalMaximum(SystemVector):
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    Not assembled by default.

    """

    _element_space: LagrangeFiniteElementSpace

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        self._element_space = element_space

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        local_maximum = np.empty(len(dof_vector))

        for dof_index in range(len(dof_vector)):
            local_maximum[dof_index] = max(
                {dof_vector[j] for j in self._element_space.dof_neighbours(dof_index)}
            )

        return local_maximum


class LocalMinimum(LocalMaximum):
    """Local maximum vector. Consider a DOF vector (ui). Then, the local maximum
    vector is defined via

        ui_max = max(uj, j is neighbour of i).

    Not assembled by default.

    """

    _element_space: LagrangeFiniteElementSpace

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        self._element_space = element_space

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        local_maximum = np.empty(len(dof_vector))

        for dof_index in range(len(dof_vector)):
            local_maximum[dof_index] = min(
                {dof_vector[j] for j in self._element_space.dof_neighbours(dof_index)}
            )

        return local_maximum
