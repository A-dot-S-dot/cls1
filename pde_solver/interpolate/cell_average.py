import numpy as np
from custom_type import ScalarFunction
from pde_solver.mesh import Mesh
from pde_solver.quadrature import GaussianQuadratureGeneralized

from .interpolator import Interpolator


class CellAverageInterpolator(Interpolator):
    """Interpolate functions by calculating averages on each cell."""

    _mesh: Mesh
    _quadrature_degree: int

    def __init__(self, mesh: Mesh, quadrature_degree: int):
        self._mesh = mesh
        self._quadrature_degree = quadrature_degree

    def interpolate(self, f: ScalarFunction) -> np.ndarray:
        dof_values = np.zeros(len(self._mesh))

        for i in range(len(dof_values)):
            dof_values[i] = self._cell_average(f, i)

        return dof_values

    def _cell_average(self, f: ScalarFunction, index: int) -> float:
        cell = self._mesh[index]
        quadrature = GaussianQuadratureGeneralized(self._quadrature_degree, cell)

        return quadrature.integrate(f) / cell.length
