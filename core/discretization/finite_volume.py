from typing import Generic, TypeVar
import numpy as np
from core.index_mapping import (
    DOFNeighbourIndicesMapping,
    LeftRightCellIndexMapping,
    LeftRightNodeIndexMapping,
)
from core.mesh import Mesh

from . import abstract

T = TypeVar("T", float, np.ndarray)


class FiniteVolumeSpace(abstract.SolverSpace, Generic[T]):
    mesh: Mesh
    left_right_cell: LeftRightCellIndexMapping
    left_right_node: LeftRightNodeIndexMapping
    dof_neighbours: DOFNeighbourIndicesMapping

    _cell_centers: np.ndarray
    _right_cell_indices: np.ndarray
    _right_cell_indices: np.ndarray

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.left_right_cell = LeftRightCellIndexMapping(mesh)
        self.left_right_node = LeftRightNodeIndexMapping(mesh)
        self.dof_neighbours = DOFNeighbourIndicesMapping(mesh, 1, self.dimension)

        self._build_cell_centers()
        self._build_left_cell_indices()
        self._build_right_cell_indices()
        self._build_left_node_indices()
        self._build_right_node_indices()

    def _build_cell_centers(self):
        self._cell_centers = np.array([(cell.a + cell.b) / 2 for cell in self.mesh])

    def _build_left_cell_indices(self):
        self._left_cell_indices = np.array(
            [self.left_right_cell(i)[0] for i in range(self.left_right_cell.dimension)]
        )

    def _build_right_cell_indices(self):
        self._right_cell_indices = np.array(
            [self.left_right_cell(i)[1] for i in range(self.left_right_cell.dimension)]
        )

    def _build_left_node_indices(self):
        self._left_node_indices = np.array(
            [self.left_right_node(i)[0] for i in range(self.left_right_node.dimension)]
        )

    def _build_right_node_indices(self):
        self._right_node_indices = np.array(
            [self.left_right_node(i)[1] for i in range(self.left_right_node.dimension)]
        )

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

    @property
    def left_cell_indices(self) -> np.ndarray:
        return self._left_cell_indices

    @property
    def right_cell_indices(self) -> np.ndarray:
        return self._right_cell_indices

    @property
    def left_node_indices(self) -> np.ndarray:
        return self._left_node_indices

    @property
    def right_node_indices(self) -> np.ndarray:
        return self._right_node_indices

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
