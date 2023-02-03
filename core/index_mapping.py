from abc import ABC, abstractmethod
from typing import List, Set, Tuple

import numpy as np


class IndexMapping(ABC):
    """Returns indices for a given set of indices."""

    @abstractmethod
    def __call__(self, *index: int) -> Tuple[int, ...]:
        ...


class NeighbourIndicesMapping(IndexMapping):
    """Returns neighboured indices for a DOF index (finite elements). It
    contains all DOF indices of the cells it belongs to.

    """

    _neighbour_indices: np.ndarray

    def _build_neighbours(self) -> List[Set[int]]:
        raise NotImplementedError

    def _build_neighbours_array(self, neighbours: List) -> np.ndarray:
        neighbours_max = self._calculate_neighbours_max(neighbours)
        neighbours = [list(n) for n in neighbours]
        neighbours = [
            self._fill(n, i, neighbours_max) for i, n in enumerate(neighbours)
        ]

        return np.array(neighbours)

    def _calculate_neighbours_max(self, neighbours: List) -> int:
        return max([len(n) for n in neighbours])

    def _fill(self, neighbours: List, index: int, number: int) -> List:
        for _ in range(number - len(neighbours)):
            neighbours.append(index)

        return neighbours

    def __call__(self, dof_index: int) -> np.ndarray:
        return np.array(list(set(self._neighbour_indices[dof_index])))

    @property
    def array(self) -> np.ndarray:
        return self._neighbour_indices


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
