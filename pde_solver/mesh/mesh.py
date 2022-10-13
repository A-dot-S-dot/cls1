from abc import ABC, abstractmethod
from typing import Iterator, Sequence

import numpy as np

from .interval import Interval


class Mesh(ABC):
    """Mesh class for coherent meshes."""

    _domain: Interval
    _cells: Sequence[Interval]

    @property
    def domain(self) -> Interval:
        return self._domain

    @property
    def step_length(self) -> float:
        simplices_step_length = [simplex.length for simplex in self._cells]
        return max(*simplices_step_length)

    @property
    @abstractmethod
    def space_steps(sefl) -> np.ndarray:
        ...

    def __iter__(self) -> Iterator[Interval]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def __getitem__(self, index: int) -> Interval:
        return self._cells[index]

    def __eq__(self, other) -> bool:
        return self._cells == other._cells

    def index(self, cell: Interval) -> int:
        return self._cells.index(cell)

    def find_cell_indices(self, point: float) -> Sequence[int]:
        indices = []
        for cell in self._cells:
            if point in cell:
                indices.append(self.index(cell))

        return indices

    def find_cells(self, point: float) -> Sequence[Interval]:
        return [self[index] for index in self.find_cell_indices(point)]

    @abstractmethod
    def refine(self) -> "Mesh":
        ...

    @abstractmethod
    def coarsen(self, coarsening_degree: int) -> "Mesh":
        ...
