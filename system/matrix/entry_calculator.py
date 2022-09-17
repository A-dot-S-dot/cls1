"""This module provides the interface and some implementations for objects which
calculate entries of system matrices."""
from abc import ABC, abstractmethod

from fem import FiniteElementSpace
from fem.lagrange import LocalLagrangeBasis
from quadrature.local import LocalElementQuadrature


class SystemMatrixEntryCalculator(ABC):
    """Object for calculating a matrix entry."""

    @abstractmethod
    def __call__(
        self, simplex_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        ...


class QuadratureBasedEntryCalculator(SystemMatrixEntryCalculator):
    """An Entry Calculator which entries are integrals.

    It's call method must be implemented by subclasses.
    """

    _local_quadrature: LocalElementQuadrature
    _local_basis: LocalLagrangeBasis

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
        self._local_basis = element_space.local_basis
