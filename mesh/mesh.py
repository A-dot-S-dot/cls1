from typing import Iterator, List, Sequence
from abc import ABC, abstractmethod

import numpy as np

from .interval import Interval


class Mesh(ABC):
    """Mesh class for coherent meshes."""

    _domain: Interval
    _grid: np.ndarray
    _simplices: Sequence[Interval]

    @property
    def domain(self) -> Interval:
        return self._domain

    @property
    def step_length(self) -> float:
        simplices_step_length = [simplex.length for simplex in self._simplices]
        return max(*simplices_step_length)

    def __iter__(self) -> Iterator[Interval]:
        return iter(self._simplices)

    def __len__(self) -> int:
        return len(self._grid) - 1

    def __getitem__(self, index: int) -> Interval:
        return self._simplices[index]

    def __eq__(self, other) -> bool:
        return self._simplices == other._simplices

    def index(self, simplex: Interval) -> int:
        return self._simplices.index(simplex)

    def find_simplex_indices(self, point: float) -> List[int]:
        indices = []
        for simplex in self._simplices:
            if point in simplex:
                indices.append(self.index(simplex))

        return indices

    def find_simplices(self, point: float) -> List[Interval]:
        return [self[index] for index in self.find_simplex_indices(point)]

    @abstractmethod
    def refine(self) -> "Mesh":
        ...
