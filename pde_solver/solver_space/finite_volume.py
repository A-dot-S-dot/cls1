import numpy as np
from pde_solver.index_mapping.finite_volume import (
    LeftRightCellIndexMapping,
    LeftRightEdgeIndexMapping,
)
from pde_solver.mesh import Mesh

from .solver_space import SolverSpace


class FiniteVolumeSpace(SolverSpace):
    periodic = True
    mesh: Mesh
    left_right_cell: LeftRightCellIndexMapping
    left_right_edge: LeftRightEdgeIndexMapping

    _cell_centers: np.ndarray

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.left_right_cell = LeftRightCellIndexMapping(mesh)
        self.left_right_edge = LeftRightEdgeIndexMapping(mesh)
        self._build_cell_centers()

    def _build_cell_centers(self):
        self._cell_centers = np.array([(cell.a + cell.b) / 2 for cell in self.mesh])

    @property
    def dimension(self):
        return len(self.mesh)

    @property
    def edge_number(self):
        return self.dimension

    @property
    def cell_centers(self) -> np.ndarray:
        return self._cell_centers


class CoarsenedFiniteVolumeSpace(FiniteVolumeSpace):
    def __init__(self, volume_space: FiniteVolumeSpace, coarsening_degree: int):
        coarsened_mesh = volume_space.mesh.coarsen(coarsening_degree)
        FiniteVolumeSpace.__init__(self, coarsened_mesh)
