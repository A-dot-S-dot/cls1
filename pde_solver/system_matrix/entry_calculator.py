"""This module provides the interface and some implementations for objects which
calculate entries of system matrices."""
from abc import ABC, abstractmethod

from pde_solver.quadrature.local import LocalElementQuadrature
from pde_solver.discretization.finite_element import LocalLagrangeBasis


class SystemMatrixEntryCalculator(ABC):
    """Object for calculating a matrix entry."""

    @abstractmethod
    def __call__(
        self, cell_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        ...


class QuadratureBasedEntryCalculator(SystemMatrixEntryCalculator):
    """An Entry Calculator which entries are integrals.

    It's call method must be implemented by subclasses.
    """

    _local_quadrature: LocalElementQuadrature
    _local_basis: LocalLagrangeBasis

    def __init__(self, polynomial_degree: int, quadrature_degree: int):
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
        self._local_basis = LocalLagrangeBasis(polynomial_degree)
