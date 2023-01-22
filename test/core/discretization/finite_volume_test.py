from unittest import TestCase

from core.discretization import finite_volume
from core.mesh import Interval, UniformMesh


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

    def test_left_cell_indices(self):
        expected_left_cell_indices = [3, 0, 1, 2]
        self.assertListEqual(
            list(self.volume_space.left_cell_indices), expected_left_cell_indices
        )

    def test_right_cell_indices(self):
        expected_right_cell_indices = [0, 1, 2, 3]
        self.assertListEqual(
            list(self.volume_space.right_cell_indices), expected_right_cell_indices
        )

    def test_left_node_indices(self):
        expected_left_node_indices = [0, 1, 2, 3]
        self.assertListEqual(
            list(self.volume_space.left_node_indices), expected_left_node_indices
        )

    def test_right_node_indices(self):
        expected_right_node_indices = [1, 2, 3, 0]
        self.assertListEqual(
            list(self.volume_space.right_node_indices), expected_right_node_indices
        )
