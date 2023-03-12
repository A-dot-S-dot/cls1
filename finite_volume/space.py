from typing import Generic, TypeVar

import numpy as np
import core

T = TypeVar("T", float, np.ndarray)


class FiniteVolumeSpace(core.SolverSpace, Generic[T]):
    mesh: core.Mesh
    cell_centers: np.ndarray

    def __init__(self, mesh: core.Mesh):
        self.mesh = mesh

        self._build_cell_centers()

    def _build_cell_centers(self):
        self.cell_centers = np.array([(cell.a + cell.b) / 2 for cell in self.mesh])

    @property
    def dimension(self):
        return len(self.mesh)

    @property
    def node_number(self):
        return self.dimension

    @property
    def grid(self) -> np.ndarray:
        return self.cell_centers

    def element(self, dof_vector: np.ndarray) -> core.CellDependentFunction[T]:
        return FiniteVolumeElement(self, dof_vector)

    def refine(self, refine_degree: int) -> "FiniteVolumeSpace":
        return FiniteVolumeSpace(self.mesh.refine(refine_degree))

    def coarsen(self, coarsening_degree: int) -> "FiniteVolumeSpace":
        return FiniteVolumeSpace(self.mesh.coarsen(coarsening_degree))

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(mesh={self.mesh})"


class FiniteVolumeElement(core.CellDependentFunction, Generic[T]):
    def __init__(self, solver_space: FiniteVolumeSpace, dof_vector: np.ndarray):
        self.space = solver_space
        self.dof_vector = dof_vector

    def __call__(self, cell_index: int, x: float) -> T:
        return self.dof_vector[cell_index]

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(dof={self.dof_vector})"


def get_finite_volume_solution(
    benchmark: core.Benchmark, mesh_size: int, save_history=False
) -> core.DiscreteSolution[FiniteVolumeSpace]:
    mesh = core.UniformMesh(benchmark.domain, mesh_size)
    space = FiniteVolumeSpace(mesh)
    interpolator = core.CellAverageInterpolator(mesh, 2)
    solution_type = (
        core.DiscreteSolutionWithHistory if save_history else core.DiscreteSolution
    )

    return solution_type(
        interpolator.interpolate(benchmark.initial_data),
        initial_time=benchmark.start_time,
        space=space,
    )
