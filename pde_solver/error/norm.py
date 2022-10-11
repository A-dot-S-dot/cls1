"""This module provides objects for calculating errors with finite elements."""
from abc import ABC, abstractmethod

import numpy as np
from custom_type import ScalarFunction
from pde_solver.mesh import AffineTransformation, Interval, Mesh

from pde_solver.quadrature import LocalElementQuadrature


class Norm(ABC):
    name: str
    mesh: Mesh

    @abstractmethod
    def __call__(self, function: ScalarFunction) -> float:
        ...


class IntegralNorm(Norm):
    _local_quadrature: LocalElementQuadrature
    _affine_mapping: AffineTransformation

    def __init__(self, quadrature_degree: int):
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
        self._affine_mapping = AffineTransformation()


class L2Norm(IntegralNorm):
    name = "L2"

    def __call__(self, function: ScalarFunction) -> float:
        integral = 0

        for cell in self.mesh:
            integral += self._calculate_norm_on_cell(function, cell)

        return np.sqrt(integral)

    def _calculate_norm_on_cell(
        self, function: ScalarFunction, cell: Interval
    ) -> float:
        node_values = np.array(
            [
                function(self._affine_mapping(node, cell)) ** 2
                for node in self._local_quadrature.nodes
            ]
        )
        return self._affine_mapping.derivative(cell) * np.dot(
            self._local_quadrature.weights, node_values
        )


class L1Norm(IntegralNorm):
    name = "L1"

    def __call__(self, function: ScalarFunction) -> float:
        integral = 0

        for cell in self.mesh:
            integral += self._calculate_norm_on_cell(function, cell)

        return integral

    def _calculate_norm_on_cell(
        self, function: ScalarFunction, cell: Interval
    ) -> float:
        node_values = np.array(
            [
                np.absolute(function(self._affine_mapping(node, cell)))
                for node in self._local_quadrature.nodes
            ]
        )
        return self._affine_mapping.derivative(cell) * np.dot(
            self._local_quadrature.weights, node_values
        )


class LInfinityNorm(Norm):
    name = "Linf"

    _points_per_cell: int

    def __init__(self, points_per_cell: int):
        self._points_per_cell = points_per_cell

    def __call__(self, function: ScalarFunction) -> float:
        maximum_per_cell = np.zeros(len(self.mesh))

        for cell_index, cell in enumerate(self.mesh):
            maximum_per_cell[cell_index] = self._calculate_norm_on_cell(function, cell)

        return float(np.amax(maximum_per_cell))

    def _calculate_norm_on_cell(
        self, function: ScalarFunction, cell: Interval
    ) -> float:
        return np.amax(
            [function(x) for x in np.linspace(cell.a, cell.b, self._points_per_cell)]
        )
