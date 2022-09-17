"""This modules provides classes of finite elements on finite element spaces."""

import numpy as np

from .abstracts import FiniteElement, FiniteElementSpace


class GlobalFiniteElement(FiniteElement):
    """Finite element which is defined by coefficients each belonging to a basis
    element of the finite element space."""

    _element_space: FiniteElementSpace
    _dof_vector: np.ndarray

    def __init__(self, element_space: FiniteElementSpace, dof_vector: np.ndarray):
        if not element_space.is_dof_vector(dof_vector):
            raise ValueError(f"{dof_vector} is not dof vector of `element_space`")

        self._dof_vector = dof_vector
        self._element_space = element_space

    def __call__(self, point: float) -> float:
        return self._element_space.get_value(point, self._dof_vector)

    def value_on_simplex(self, point: float, simplex_index: int) -> float:
        return self._element_space.get_value_on_simplex(
            point, self._dof_vector, simplex_index
        )

    def derivative(self, point: float) -> float:
        return self._element_space.get_derivative(point, self._dof_vector)

    def derivative_on_simplex(self, point: float, simplex_index: int) -> float:
        return self._element_space.get_derivative_on_simplex(
            point, self._dof_vector, simplex_index
        )

    @property
    def degree_of_freedom(self) -> int:
        return len(self._dof_vector)
