from typing import Generic, TypeVar

import numpy as np
from core.benchmark import Benchmark
from core.discrete_solution import DiscreteSolution, DiscreteSolutionWithHistory
from .index_mapping import NeighbourIndicesMapping
from core.interpolate import CellAverageInterpolator
from core.mesh import Mesh, UniformMesh
from core.space import CellDependentFunction, SolverSpace

T = TypeVar("T", float, np.ndarray)


class FiniteVolumeSpace(SolverSpace, Generic[T]):
    mesh: Mesh
    dof_neighbours: NeighbourIndicesMapping

    _cell_centers: np.ndarray

    def __init__(self, mesh: Mesh, periodic=False):
        self.mesh = mesh
        self.dof_neighbours = NeighbourIndicesMapping(len(mesh), periodic)

        self._build_cell_centers()

    def _build_cell_centers(self):
        self._cell_centers = np.array([(cell.a + cell.b) / 2 for cell in self.mesh])

    @property
    def dimension(self):
        return len(self.mesh)

    @property
    def node_number(self):
        return self.dimension

    @property
    def cell_centers(self) -> np.ndarray:
        return self._cell_centers

    @property
    def grid(self) -> np.ndarray:
        return self.cell_centers

    def element(self, dof_vector: np.ndarray) -> CellDependentFunction[T]:
        return FiniteVolumeElement(self, dof_vector)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(mesh={self.mesh})"


class FiniteVolumeElement(CellDependentFunction, Generic[T]):
    def __init__(self, solver_space: FiniteVolumeSpace, dof_vector: np.ndarray):
        self._solver_space = solver_space
        self._dof_vector = dof_vector

    def __call__(self, cell_index: int, x: float) -> T:
        return self._dof_vector[cell_index]


def build_finite_volume_solution(
    benchmark: Benchmark,
    mesh_size: int,
    save_history=False,
    periodic=False,
) -> DiscreteSolution[FiniteVolumeSpace]:
    mesh = UniformMesh(benchmark.domain, mesh_size)
    space = FiniteVolumeSpace(mesh, periodic=periodic)
    interpolator = CellAverageInterpolator(mesh, 2)
    solution_type = DiscreteSolutionWithHistory if save_history else DiscreteSolution

    return solution_type(
        interpolator.interpolate(benchmark.initial_data),
        start_time=benchmark.start_time,
        space=space,
    )
