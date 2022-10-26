from abc import ABC, abstractmethod
from typing import Sequence, Callable

import numpy as np
from pde_solver.mesh import Mesh
from pde_solver.quadrature import GaussianQuadrature

ScalarFunction = Callable[[float], float]


class Interpolator(ABC):
    """Interpolates scalar functions by returning DOF Vectors."""

    def interpolate(self, function) -> np.ndarray:
        if isinstance(function(0), float) or isinstance(function(0), int):
            return self._interpolate_scalar(function)
        else:
            return np.array(
                [
                    self._interpolate_scalar(lambda x: function(x)[i])
                    for i in range(len(function(0)))
                ]
            ).T

    @abstractmethod
    def _interpolate_scalar(self, function) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class CellAverageInterpolator(Interpolator):
    """Interpolate functions by calculating averages on each cell."""

    _mesh: Mesh
    _quadrature_degree: int

    def __init__(self, mesh: Mesh, quadrature_degree: int):
        self._mesh = mesh
        self._quadrature_degree = quadrature_degree

    def _interpolate_scalar(self, function) -> np.ndarray:
        dof_values = np.zeros(len(self._mesh))

        for i in range(len(dof_values)):
            dof_values[i] = self._cell_average(function, i)

        return dof_values

    def _cell_average(self, function, index: int) -> float:
        cell = self._mesh[index]
        quadrature = GaussianQuadrature(self._quadrature_degree, cell)

        return quadrature.integrate(function) / cell.length


class NodeValuesInterpolator(Interpolator):
    """Interpolate functions by calculating values on given nodes."""

    _nodes: Sequence[float]

    def __init__(self, *nodes: float):
        self._nodes = nodes

    def _interpolate_scalar(self, f: ScalarFunction) -> np.ndarray:
        return np.array([f(node) for node in self._nodes])
