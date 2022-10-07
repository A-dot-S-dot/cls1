"""This modules provides classes of finite elements on finite element spaces."""

from typing import Sequence

import numpy as np
from pde_solver.mesh import AffineTransformation, Mesh
from pde_solver.solver_space import LagrangeFiniteElementSpace

from pde_solver.discretization.local_lagrange import (
    LOCAL_LAGRANGE_BASIS,
    LocalLagrangeBasis,
)


class LagrangeFiniteElement:
    """Finite element which is defined by coefficients each belonging to a basis
    element of the finite element space."""

    _element_space: LagrangeFiniteElementSpace
    _mesh: Mesh
    _dof_vector: np.ndarray
    _local_basis: LocalLagrangeBasis
    _affine_transformation: AffineTransformation

    def __init__(
        self, element_space: LagrangeFiniteElementSpace, dof_vector: np.ndarray
    ):
        self._element_space = element_space
        self._mesh = element_space.mesh
        self._dof_vector = dof_vector
        self._local_basis = LOCAL_LAGRANGE_BASIS[element_space.polynomial_degree]
        self._affine_transformation = AffineTransformation()

    def __call__(self, point: float) -> float:
        cell_index = self._mesh.find_cell_indices(point)[0]
        return self._get_value_on_cell(point, cell_index)

    def _get_value_on_cell(self, point: float, cell_index: int) -> float:
        cell = self._mesh[cell_index]
        value = 0

        for local_index, local_element in enumerate(self._local_basis):
            global_index = self._element_space.global_index(cell_index, local_index)
            value += self._dof_vector[global_index] * local_element(
                self._affine_transformation.inverse(point, cell)
            )

        return value

    def derivative(self, point: float) -> float:
        cell_indices = self._mesh.find_cell_indices(point)

        if len(cell_indices) == 1:
            return self._get_derivative_on_cell(point, *cell_indices)
        elif len(cell_indices) == 2:
            return self._get_derivative_on_edge(point, cell_indices)
        else:
            raise ValueError

    def _get_derivative_on_cell(self, point: float, cell_index: int) -> float:
        cell = self._mesh[cell_index]
        value = 0

        for local_index, local_element in enumerate(self._local_basis):
            global_index = self._element_space.global_index(cell_index, local_index)
            local_derivative = np.array(
                local_element.derivative(
                    self._affine_transformation.inverse(point, cell)
                )
            )
            value += self._dof_vector[global_index] * local_derivative

        return self._affine_transformation.inverse_derivative(cell) * value

    def _get_derivative_on_edge(
        self, point: float, cell_indices: Sequence[int]
    ) -> float:
        # derivatives at boundary of an simplex are not defined in general
        derivative_1 = self._get_derivative_on_cell(point, cell_indices[0])
        derivative_2 = self._get_derivative_on_cell(point, cell_indices[1])

        if derivative_1 == derivative_2:
            return derivative_1
        else:
            raise ValueError("Derivative on edge not defined.")
