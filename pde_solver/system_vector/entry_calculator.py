from abc import ABC, abstractmethod

from pde_solver.solver_space import LagrangeFiniteElementSpace
from pde_solver.quadrature import LocalElementQuadrature


class VectorEntryCalculator(ABC):
    """Class for calculating a vector entry using local to global principles."""

    @abstractmethod
    def __call__(self, cell_index: int, local_index: int) -> float:
        ...


class QuadratureBasedEntryCalculator(VectorEntryCalculator):
    """An Entry Calculator which entries are integrals.

    It's call method must be implemented by subclasses.
    """

    _local_quadrature: LocalElementQuadrature

    def __init__(
        self, element_space: LagrangeFiniteElementSpace, quadrature_degree: int
    ):
        self._element_space = element_space
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
