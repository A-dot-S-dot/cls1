from abc import ABC, abstractmethod

from mesh import Mesh


class DOFIndexMapping(ABC):
    _output_dimension: int

    @abstractmethod
    def __call__(self, simplex_index: int, local_index: int) -> int:
        ...

    @property
    def output_dimension(self) -> int:
        return self._output_dimension


class PeriodicDOFIndexMapping(DOFIndexMapping):
    _mesh: Mesh
    _last_simplex_index: int
    _last_local_index: int

    def __init__(self, mesh: Mesh, local_basis_length: int):
        self._mesh = mesh
        self._last_local_index = local_basis_length - 1
        self._last_simplex_index = len(mesh) - 1
        self._output_dimension = len(mesh) * (local_basis_length - 1)

    def __call__(self, simplex_index: int, local_index: int) -> int:
        if (
            simplex_index == self._last_simplex_index
            and local_index == self._last_local_index
        ):
            return 0
        else:
            return simplex_index * self._last_local_index + local_index
