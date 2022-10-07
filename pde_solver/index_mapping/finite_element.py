from pde_solver.mesh import Mesh
from typing import Tuple, Set, Sequence

from .index_mapping import IndexMapping


class GlobalIndexMapping(IndexMapping):
    """Returns the global index. Periodic boundaries assumed."""

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
    """Returns neighboured indices for a DOF index. It contains all DOF indices
    of the simplices it belongs to.

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
