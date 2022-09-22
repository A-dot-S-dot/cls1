"""This module provides abstract classes for finite elements."""
from abc import ABC, abstractmethod
from typing import Iterator, Sequence, Set

import numpy as np
from math_type import FunctionRealToReal
from mesh import Interval, Mesh


class FiniteElement(ABC):
    """Abstract class representing a finite element"""

    @abstractmethod
    def __call__(self, point: float) -> float:
        ...

    @abstractmethod
    def derivative(self, coordinate: float) -> float:
        ...


class LocalFiniteElement(FiniteElement):
    """Finite element basis element on the standard simplex."""

    _call_method: FunctionRealToReal
    _derivative: FunctionRealToReal

    def __init__(
        self,
        call_method: FunctionRealToReal,
        derivative: FunctionRealToReal,
    ):
        self._call_method = call_method
        self._derivative = derivative

    def __call__(self, point: float) -> float:
        return self._call_method(point)

    def derivative(self, point: float) -> float:
        return self._derivative(point)


class FiniteElementBasis(ABC):
    """Abstract class for Finite element basis."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[FiniteElement]:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> FiniteElement:
        ...


class LocalFiniteElementBasis(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def nodes(self) -> Sequence[float]:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[LocalFiniteElement]:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> LocalFiniteElement:
        ...


class FiniteElementBasisContainer(ABC):
    """Container class for creating finite element basis for a given polynomial degree."""

    @abstractmethod
    def __getitem__(self, polynomial_degree: int) -> FiniteElementBasis:
        ...


class FiniteElementSpace(ABC):
    """Abstract class for Finite Element space."""

    def is_dof_vector(self, dof_vector: np.ndarray) -> bool:
        """Checks if the length of the vectors matches the dimension of the element space."""
        return len(dof_vector) == self.dimension

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def polynomial_degree(self) -> int:
        ...

    @property
    @abstractmethod
    def domain(self) -> Interval:
        ...

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        ...

    @property
    @abstractmethod
    def indices_per_simplex(self) -> int:
        ...

    @property
    @abstractmethod
    def local_basis(self) -> LocalFiniteElementBasis:
        ...

    @abstractmethod
    def get_global_index(self, simplex_index: int, local_index: int) -> int:
        ...

    @abstractmethod
    def get_neighbour_indices(self, index: int) -> Sequence[int]:
        ...

    @abstractmethod
    def get_value(self, point: float, dof_vector: np.ndarray) -> float:
        """Returns

        sum_i(DOF_VECTOR_i phi_i(POINT)),

        where phi_i denotes the basis of the finite element space.
        """
        ...

    @abstractmethod
    def get_value_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        ...

    @abstractmethod
    def get_derivative(self, point: float, dof_vector: np.ndarray) -> float:
        """Returns

        sum_i(DOF_VECTOR_i grad phi_i(POINT)),

        where phi_i denotes the basis of the finite element space.
        """
        ...

    @abstractmethod
    def get_derivative_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        ...

    @abstractmethod
    def interpolate(self, f: FunctionRealToReal) -> np.ndarray:
        ...

    def __eq__(self, other) -> bool:
        return (
            self.mesh == other.mesh
            and self.polynomial_degree == other.polynomial_degree
        )
