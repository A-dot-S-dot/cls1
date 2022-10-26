from typing import Generic, TypeVar
import numpy as np
from pde_solver.index_mapping import (
    LeftRightCellIndexMapping,
    LeftRightNodeIndexMapping,
)
from pde_solver.mesh import Mesh

from . import abstract

T = TypeVar("T", float, np.ndarray)


class FiniteVolumeSpace(abstract.SolverSpace, Generic[T]):
    mesh: Mesh
    left_right_cell: LeftRightCellIndexMapping
    left_right_edge: LeftRightNodeIndexMapping

    _cell_centers: np.ndarray

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.left_right_cell = LeftRightCellIndexMapping(mesh)
        self.left_right_edge = LeftRightNodeIndexMapping(mesh)
        self._build_cell_centers()

    def _build_cell_centers(self):
        self._cell_centers = np.array([(cell.a + cell.b) / 2 for cell in self.mesh])

    @property
    def dimension(self):
        return len(self.mesh)

    @property
    def node_number(self):
        return self.dimension

    @property
    def cell_centers(self) -> np.ndarray:
        return self._cell_centers

    @property
    def grid(self) -> np.ndarray:
        return self.cell_centers

    def element(self, dof_vector: np.ndarray) -> abstract.CellDependentFunction[T]:
        return FiniteVolumeElement(self, dof_vector)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(mesh={self.mesh})"


class FiniteVolumeElement(abstract.CellDependentFunction, Generic[T]):
    def __init__(self, solver_space: FiniteVolumeSpace, dof_vector: np.ndarray):
        self._solver_space = solver_space
        self._dof_vector = dof_vector

    def __call__(self, cell_index: int, x: float) -> T:
        return self._dof_vector[cell_index]
