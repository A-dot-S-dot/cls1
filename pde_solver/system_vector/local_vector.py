import numpy as np
from pde_solver.discretization.finite_element import LagrangeFiniteElementSpace

from .entry_calculator import VectorEntryCalculator
from .system_vector import SystemVector


class LocallyAssembledVector(SystemVector):
    """Assembles with 'local to global' principles."""

    _element_space: LagrangeFiniteElementSpace
    _entry_calculator: VectorEntryCalculator

    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        entry_calculator: VectorEntryCalculator,
    ):
        self._element_space = element_space
        self._entry_calculator = entry_calculator

    def __call__(self) -> np.ndarray:
        vector = np.zeros(self._element_space.dimension)

        for cell_index in range(len(self._element_space.mesh)):
            for local_index in range(self._element_space.polynomial_degree + 1):
                global_index = self._element_space.global_index(cell_index, local_index)

                vector[global_index] += self._entry_calculator(cell_index, local_index)

        return vector
