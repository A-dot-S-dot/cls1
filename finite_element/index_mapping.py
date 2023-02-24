from typing import List, Set

import core


class GlobalIndexMapping(core.IndexMapping):
    """Returns the global index for given cell and a local index (Finite
    Elements). Periodic boundaries assumed.

    """

    _last_cell_index: int
    _indices_per_cell: int

    def __init__(self, mesh: core.Mesh, polynomial_degree: int, periodic=False):
        assert polynomial_degree > 0
        self._polynomial_degree = polynomial_degree

        self._last_cell_index = len(mesh) - 1
        self._indices_per_cell = polynomial_degree + 1
        self._periodic = periodic

    @property
    def periodic(self) -> bool:
        return self._periodic

    @property
    def polynomial_degree(self) -> int:
        return self._polynomial_degree

    @property
    def indices_per_cell(self) -> int:
        return self._indices_per_cell

    @property
    def mesh_size(self) -> int:
        return self._last_cell_index + 1

    @property
    def dimension(self) -> int:
        nodes = self.polynomial_degree * self.mesh_size
        return nodes if self.periodic else nodes + 1

    def __call__(self, cell_index: int, local_index: int) -> int:
        assert local_index + 1 <= self.indices_per_cell

        if (
            cell_index == self._last_cell_index
            and local_index + 1 == self._indices_per_cell
            and self.periodic
        ):
            return 0
        else:
            return cell_index * self.polynomial_degree + local_index


class NeighbourIndicesMapping(core.NeighbourIndicesMapping):
    """Returns neighboured indices for a DOF index (finite elements). It
    contains all DOF indices of the cells it belongs to.

    """

    _global_index: GlobalIndexMapping

    def __init__(self, global_index: GlobalIndexMapping):
        self._global_index = global_index
        neighbours = self._build_neighbours()
        self._neighbour_indices = self._build_neighbours_array(neighbours)

    def _build_neighbours(self) -> List[Set[int]]:
        neighbour_indices = [set() for _ in range(self._global_index.dimension)]

        for cell_index in range(self._global_index.mesh_size):
            for local_index_1 in range(self._global_index.indices_per_cell):
                global_index_1 = self._global_index(cell_index, local_index_1)
                for local_index_2 in range(self._global_index.indices_per_cell):
                    global_index_2 = self._global_index(cell_index, local_index_2)
                    neighbour_indices[global_index_1] |= {global_index_2}

        return neighbour_indices
