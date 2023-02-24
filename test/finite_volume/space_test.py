from unittest import TestCase

from core.mesh import Interval, UniformMesh

import finite_volume


class TestFiniteVolumeSpace(TestCase):
    mesh_size = 4
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, mesh_size)
    volume_space = finite_volume.FiniteVolumeSpace(mesh)

    def test_basis_nodes(self):
        expected_cell_centers = [1 / 8, 3 / 8, 5 / 8, 7 / 8]
        self.assertListEqual(
            list(self.volume_space.cell_centers), expected_cell_centers
        )
