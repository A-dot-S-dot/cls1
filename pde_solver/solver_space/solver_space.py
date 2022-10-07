from abc import abstractmethod, ABC

from pde_solver.mesh import Mesh


class SolverSpace(ABC):
    mesh: Mesh

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...
