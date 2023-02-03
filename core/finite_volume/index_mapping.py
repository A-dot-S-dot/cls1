from typing import List, Set

from .. import index_mapping


class NeighbourIndicesMapping(index_mapping.NeighbourIndicesMapping):
    """Returns neighboured indices for a DOF index (finite elements). It
    contains all DOF indices of the cells it belongs to.

    """

    _mesh_size: int
    _periodic: bool

    def __init__(self, mesh_size: int, periodic=False):
        self._mesh_size = mesh_size
        self._periodic = periodic

        neighbours = self._build_neighbours()
        self._neighbour_indices = self._build_neighbours_array(neighbours)

    def _build_neighbours(self) -> List[Set[int]]:
        neighbour_indices = [set([i - 1, i, i + 1]) for i in range(self._mesh_size)]
        neighbour_indices[0] -= set([-1])
        neighbour_indices[-1] -= set([self._mesh_size])

        if self._periodic:
            neighbour_indices[0] |= set([self._mesh_size - 1])
            neighbour_indices[-1] |= set([0])

        return neighbour_indices
