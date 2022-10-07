from unittest import TestCase

from pde_solver.mesh import Interval, UniformMesh
from pde_solver.solver_space import FiniteVolumeSpace


class TestFiniteVolumeSpace(TestCase):
    mesh_size = 2
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, mesh_size)
    volume_space = FiniteVolumeSpace(mesh)
    expected_cell_centers = [0.25, 0.75]

    def test_basis_nodes(self):
        self.assertListEqual(
            list(self.volume_space.cell_centers), self.expected_cell_centers
        )
