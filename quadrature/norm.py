"""This module provides objects for calculating errors with finite elements."""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from mesh import Interval, Mesh
from mesh.uniform import UniformMesh
from mesh.transformation import AffineTransformation

from quadrature.local import LocalElementQuadrature

ScalarFunction = Callable[[float], float]


class Norm(ABC):
    _mesh: Mesh
    _name: str

    @abstractmethod
    def set_mesh(self, mesh: Mesh):
        ...

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self, function: ScalarFunction) -> float:
        ...


class MeshDependentIntegralNorm(Norm):
    _mesh: Mesh
    _name: str
    _local_quadrature: LocalElementQuadrature
    _affine_mapping: AffineTransformation
    _determinant_derivative_affine_mapping: float

    def __init__(self, mesh: Mesh, quadrature_degree: int):
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
        self._affine_mapping = AffineTransformation()

        self.set_mesh(mesh)

    def set_mesh(self, mesh: Mesh):
        if not isinstance(mesh, UniformMesh):
            raise NotImplementedError(
                "Integral norm not implemented for not uniform meshes"
            )

        self._mesh = mesh
        self._determinant_derivative_affine_mapping = mesh.step_length


class L2Norm(MeshDependentIntegralNorm):
    _name = "L2-Norm"

    def __call__(self, function: ScalarFunction) -> float:
        integral = 0

        for simplex in self._mesh:
            integral += self._calculate_norm_on_simplex(function, simplex)

        return np.sqrt(self._determinant_derivative_affine_mapping * integral)

    def _calculate_norm_on_simplex(
        self, function: ScalarFunction, simplex: Interval
    ) -> float:
        node_values = np.array(
            [
                function(self._affine_mapping(node, simplex)) ** 2
                for node in self._local_quadrature.nodes
            ]
        )
        return np.dot(self._local_quadrature.weights, node_values)


class L1Norm(MeshDependentIntegralNorm):
    _name = "L1-Norm"

    def __call__(self, function: ScalarFunction) -> float:
        integral = 0

        for simplex in self._mesh:
            integral += self._calculate_norm_on_simplex(function, simplex)

        return self._determinant_derivative_affine_mapping * integral

    def _calculate_norm_on_simplex(
        self, function: ScalarFunction, simplex: Interval
    ) -> float:
        node_values = np.array(
            [
                np.absolute(function(self._affine_mapping(node, simplex)))
                for node in self._local_quadrature.nodes
            ]
        )
        return np.dot(self._local_quadrature.weights, node_values)


class LInfinityNorm(Norm):
    _mesh: Mesh
    _name = "Linf-Norm"
    _points_per_simplex: int

    def __init__(self, mesh: Mesh, points_per_simplex: int):
        self._points_per_simplex = points_per_simplex
        self.set_mesh(mesh)

    def set_mesh(self, mesh: Mesh):
        self._mesh = mesh

    def __call__(self, function: ScalarFunction) -> float:
        maximum_per_simplex = np.zeros(len(self._mesh))

        for simplex_index, simplex in enumerate(self._mesh):
            maximum_per_simplex[simplex_index] = self._calculate_norm_on_simplex(
                function, simplex
            )

        return float(np.amax(maximum_per_simplex))

    def _calculate_norm_on_simplex(
        self, function: ScalarFunction, simplex: Interval
    ) -> float:
        return np.amax(
            [
                function(x)
                for x in np.linspace(simplex.a, simplex.b, self._points_per_simplex)
            ]
        )
