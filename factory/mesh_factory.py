from abc import ABC, abstractmethod

from mesh import Interval
from mesh.uniform import UniformMesh


class MeshFactory(ABC):
    @abstractmethod
    def get_mesh(self):
        ...


class UniformMeshFactory(MeshFactory):
    _domain: Interval
    _elements_number: int

    def __init__(self, domain: Interval, elements_number: int):
        self._domain = domain
        self._elements_number = elements_number

    def get_mesh(self) -> UniformMesh:
        return UniformMesh(self._domain, self._elements_number)
