from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from .index_mapping import (
    DOFNeighbourIndicesMapping,
)
from .mesh import Mesh


T = TypeVar("T", float, np.ndarray)


class FastFunction(ABC):
    """Returns a value or derivative for certain cell indices of a mesh and a
    point which is also accessed by an index.

    """

    @abstractmethod
    def __call__(self, cell_index: int, local_index: int):
        ...

    @abstractmethod
    def derivative(self, cell_index: int, local_index: int):
        ...


class CellDependentFunction(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, cell_index: int, x: float) -> T:
        ...

    def derivative(self, cell_index: int, x: float) -> T:
        raise NotImplementedError


class SolverSpace(ABC, Generic[T]):
    mesh: Mesh
    dof_neighbours: DOFNeighbourIndicesMapping

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def grid(self) -> np.ndarray:
        ...

    @abstractmethod
    def element(self, dof_vector: np.ndarray) -> CellDependentFunction[T]:
        ...


class EmptySpace(SolverSpace):
    @property
    def dimension(self) -> int:
        return 0

    @property
    def grid(self) -> np.ndarray:
        return np.empty(0)

    def element(self, dof_vector: np.ndarray):
        ...
