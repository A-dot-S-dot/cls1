from typing import Tuple

from pde_solver.mesh import Mesh

from .index_mapping import IndexMapping


class LeftRightCellIndexMapping(IndexMapping):
    """Returns the left and right cell index of an edge. Periodic boundaries assumed."""

    _dimension: int

    def __init__(self, mesh: Mesh):
        self._dimension = len(mesh)

    def __call__(self, edge_index: int) -> Tuple[int, int]:
        right_index = edge_index
        if right_index == 0:
            left_index = self._dimension - 1
        else:
            left_index = edge_index - 1

        return (left_index, right_index)


class LeftRightEdgeIndexMapping(IndexMapping):
    """Returns the left and right edge index of a cell. Periodic boundaries assumed."""

    def __init__(self, mesh: Mesh):
        self._dimension = len(mesh)

    def __call__(self, cell_index: int) -> Tuple[int, int]:
        left_index = cell_index

        if left_index == self._dimension - 1:
            right_index = 0
        else:
            right_index = cell_index + 1

        return (left_index, right_index)
