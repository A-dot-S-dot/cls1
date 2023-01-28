from abc import ABC, abstractmethod
from itertools import product
from typing import Callable, Sequence

import numpy as np

from .mesh import Mesh
from .quadrature import GaussianQuadrature


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

    def _interpolate_scalar(self, f: Callable[[float], float]) -> np.ndarray:
        return np.array([f(node) for node in self._nodes])


class TemporalInterpolator:
    """Interpolate time dependent values for a different time evolution."""

    def __call__(
        self, old_time: np.ndarray, values: np.ndarray, new_time: np.ndarray
    ) -> np.ndarray:
        interpolated_values = np.empty((len(new_time), *values[0].shape))

        for index in product(*[range(dim) for dim in values[0].shape]):
            # array[(slice(start, end))]=array[:]
            interpolated_values[(slice(0, len(new_time)), *index)] = np.interp(
                new_time,
                old_time,
                values[(slice(0, len(old_time)), *index)],
            )

        return interpolated_values


class SpatialInterpolator:
    """Interpolate time dependent values on a different grid."""

    def __call__(
        self, old_grid: np.ndarray, values: np.ndarray, new_grid: np.ndarray
    ) -> np.ndarray:
        """Returns an interpolation on the new grid.

        Note, values on new grid points x which are strictly less (or greater) than points in the old grid points xp are set to the first (or the last) value fp, i.e.

        x < xp[0] => f(x)=fp[0]

        x > xp[-1] => f(X)=fp[-1]

        """
        time_dimension = values.shape[0]
        spatial_dimension = values.shape[2:]

        interpolated_values = np.empty(
            (time_dimension, len(new_grid), *spatial_dimension)
        )

        for index in product(
            range(time_dimension), *[range(dim) for dim in spatial_dimension]
        ):
            # array[(slice(start, end))]=array[:]
            interpolated_values[
                (index[0], slice(0, len(new_grid)), *index[1:])
            ] = np.interp(
                new_grid,
                old_grid,
                values[(index[0], slice(0, len(old_grid)), *index[1:])],
            )

        return interpolated_values
