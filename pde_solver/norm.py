from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import numpy as np

from pde_solver.mesh import AffineTransformation, Mesh
from pde_solver.quadrature import LocalElementQuadrature

T = TypeVar("T", float, np.ndarray)


class Norm(ABC):
    name: str

    _mesh: Mesh

    def __init__(self, mesh: Mesh, **kwargs):
        self._mesh = mesh

    @abstractmethod
    def __call__(self, function: Callable[[int, float], T]) -> T:
        ...


class IntegralNorm(Norm):
    _local_quadrature: LocalElementQuadrature
    _affine_mapping: AffineTransformation

    def __init__(self, mesh: Mesh, quadrature_degree=None):
        Norm.__init__(self, mesh)
        self._local_quadrature = LocalElementQuadrature(quadrature_degree or 1)
        self._affine_mapping = AffineTransformation()


class L2Norm(IntegralNorm):
    name = "L2"

    def __call__(self, function: Callable[[int, float], T]) -> T:
        cell_values = np.array(
            [
                self._calculate_norm_on_cell(function, cell_index)
                for cell_index in range(len(self._mesh))
            ]
        )
        return np.sqrt(np.sum(cell_values, axis=0))

    def _calculate_norm_on_cell(
        self, function: Callable[[int, float], T], cell_index: int
    ) -> T:
        cell = self._mesh[cell_index]
        node_values = np.array(
            [
                function(cell_index, self._affine_mapping(node, cell)) ** 2
                for node in self._local_quadrature.nodes
            ]
        )
        return self._affine_mapping.derivative(cell) * np.sum(
            np.array(
                [
                    w * value
                    for w, value in zip(self._local_quadrature.weights, node_values)
                ]
            ),
            axis=0,
        )


class L1Norm(IntegralNorm):
    name = "L1"

    def __call__(self, function: Callable[[int, float], T]) -> T:
        cell_values = np.array(
            [
                self._calculate_norm_on_cell(function, cell_index)
                for cell_index in range(len(self._mesh))
            ]
        )
        return np.sum(cell_values, axis=0)

    def _calculate_norm_on_cell(
        self, function: Callable[[int, float], T], cell_index: int
    ) -> T:
        cell = self._mesh[cell_index]
        node_values = np.array(
            [
                np.absolute(function(cell_index, self._affine_mapping(node, cell)))
                for node in self._local_quadrature.nodes
            ]
        )
        return self._affine_mapping.derivative(cell) * np.sum(
            np.array(
                [
                    w * value
                    for w, value in zip(self._local_quadrature.weights, node_values)
                ]
            ),
            axis=0,
        )


class solver_spaces(Norm):
    name = "Linf"

    _points_per_cell: int

    def __init__(self, mesh: Mesh, points_per_cell=None):
        Norm.__init__(self, mesh)
        self._points_per_cell = points_per_cell or 10

    def __call__(self, function: Callable[[int, float], T]) -> T:
        cell_values = np.array(
            [
                self._calculate_norm_on_cell(function, cell_index)
                for cell_index in range(len(self._mesh))
            ]
        )
        return np.amax(cell_values, axis=0)

    def _calculate_norm_on_cell(
        self, function: Callable[[int, float], T], cell_index: int
    ) -> T:
        cell = self._mesh[cell_index]
        return np.amax(
            np.array(
                [
                    np.absolute(function(cell_index, x))
                    for x in np.linspace(cell.a, cell.b, self._points_per_cell)
                ]
            ),
            axis=0,
        )
