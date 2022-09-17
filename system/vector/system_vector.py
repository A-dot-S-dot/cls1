import numpy as np
from fem import FiniteElementSpace

from .entry_calculator import SystemVectorEntryCalculator


class SystemVector:
    _element_space: FiniteElementSpace
    _values: np.ndarray

    def __init__(self, element_space: FiniteElementSpace):
        self._values = np.zeros(element_space.dimension)
        self._element_space = element_space

    def assemble(self):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return self._element_space.dimension

    @property
    def element_space(self) -> FiniteElementSpace:
        return self._element_space

    @property
    def values(self) -> np.ndarray:
        return self._values

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __iter__(self):
        return iter(self._values)

    def __contains__(self, value):
        return value in self._values

    def __repr__(self):
        return self._values.__repr__()


class LocallyAssembledSystemVector(SystemVector):
    """Uses an assembler which assembles with 'local to global' principles."""

    _entry_calculator: SystemVectorEntryCalculator

    def __init__(
        self,
        element_space: FiniteElementSpace,
        entry_calculator: SystemVectorEntryCalculator,
    ):
        SystemVector.__init__(self, element_space)
        self._entry_calculator = entry_calculator

    def assemble(self):
        self[:] = 0

        for simplex_index in range(len(self._element_space.mesh)):
            for local_index in range(self._element_space.indices_per_simplex):
                global_index = self._element_space.get_global_index(
                    simplex_index, local_index
                )

                self[global_index] += self._entry_calculator(simplex_index, local_index)
