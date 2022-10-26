from abc import ABC, abstractmethod
from typing import Sequence, Set, Tuple

from pde_solver.mesh import Mesh


class IndexMapping(ABC):
    """Returns indices for a given set of indices."""

    @abstractmethod
    def __call__(self, *index: int) -> Tuple[int, ...]:
        ...


class FineGridIndicesMapping(IndexMapping):
    """Returns the fine cell indices belonging to a coarse one."""

    coarsening_degree: int

    def __init__(self, fine_mesh_size: int, coarsening_degree: int):
        self.coarsening_degree = coarsening_degree
        self._assert_admissible_coarsening_degree(fine_mesh_size)

    def _assert_admissible_coarsening_degree(self, mesh_size: int):
        if mesh_size % self.coarsening_degree != 0:
            raise ValueError(
                f"Mesh size of {mesh_size} is not divisible with respect to coarsening degree {self.coarsening_degree}."
            )

    def __call__(self, coarsen_cell_index: int) -> Tuple[int, ...]:
        return tuple(
            [
                self.coarsening_degree * coarsen_cell_index + i
                for i in range(self.coarsening_degree)
            ]
        )


class LeftRightCellIndexMapping(IndexMapping):
    """Returns the left and right cell index of a node. Periodic boundaries assumed."""

    _dimension: int

    def __init__(self, mesh: Mesh):
        self._dimension = len(mesh)

    def __call__(self, node_index: int) -> Tuple[int, int]:
        right_index = node_index
        if right_index == 0:
            left_index = self._dimension - 1
        else:
            left_index = node_index - 1

        return (left_index, right_index)


class LeftRightNodeIndexMapping(IndexMapping):
    """Returns the left and right node indices of a cell. Periodic boundaries assumed."""

    def __init__(self, mesh: Mesh):
        self._dimension = len(mesh)

    def __call__(self, cell_index: int) -> Tuple[int, int]:
        left_index = cell_index

        if left_index == self._dimension - 1:
            right_index = 0
        else:
            right_index = cell_index + 1

        return (left_index, right_index)


class GlobalIndexMapping(IndexMapping):
    """Returns the global index for given cell and a local index (Finite
    Elements). Periodic boundaries assumed.

    """

    periodic = True

    _last_cell_index: int
    _last_local_index: int

    def __init__(self, mesh: Mesh, polynomial_degree: int):
        self._last_cell_index = len(mesh) - 1
        self._last_local_index = polynomial_degree

    def __call__(self, cell_index: int, local_index: int) -> int:
        if (
            cell_index == self._last_cell_index
            and local_index == self._last_local_index
        ):
            return 0
        else:
            return cell_index * self._last_local_index + local_index


class DOFNeighbourIndicesMapping(IndexMapping):
    """Returns neighboured indices for a DOF index (finite elements). It
    contains all DOF indices of the cells it belongs to.

    """

    _mesh: Mesh
    _polynomial_degree: int
    _dimension: int
    _neighbour_indices: Sequence[Set[int]]

    def __init__(self, mesh: Mesh, polynomial_degree: int, dimension: int):
        self._mesh = mesh
        self._polynomial_degree = polynomial_degree
        self._dimension = dimension
        self._global_index = GlobalIndexMapping(mesh, polynomial_degree)

        self._build_neighbours()

    def _build_neighbours(self):
        self._neighbour_indices = [set() for _ in range(self._dimension)]

        for cell_index in range(len(self._mesh)):
            for local_index_1 in range(self._polynomial_degree + 1):
                global_index_1 = self._global_index(cell_index, local_index_1)
                for local_index_2 in range(self._polynomial_degree + 1):
                    global_index_2 = self._global_index(cell_index, local_index_2)
                    self._neighbour_indices[global_index_1] |= {global_index_2}

    def __call__(self, dof_index: int) -> Tuple[int, ...]:
        return tuple(self._neighbour_indices[dof_index])
