from abc import ABC, abstractmethod

from fem import FiniteElementSpace
from quadrature.local import LocalElementQuadrature


class SystemVectorEntryCalculator(ABC):
    """Class for calculating a vector entry using local to global principles."""

    @abstractmethod
    def __call__(self, simplex_index: int, local_index: int) -> float:
        ...


class QuadratureBasedEntryCalculator(SystemVectorEntryCalculator):
    """An Entry Calculator which entries are integrals.

    It's call method must be implemented by subclasses.
    """

    _element_space: FiniteElementSpace
    _local_quadrature: LocalElementQuadrature

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        self._element_space = element_space
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
