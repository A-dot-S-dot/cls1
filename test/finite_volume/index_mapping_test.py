from test.test_helper import LINEAR_MESH
from unittest import TestCase

from finite_volume.index_mapping import *


class TestDOFNeighbourIndicesMapping(TestCase):
    def test_non_periodic_boundaries(self):
        test_dof_indices = [0, 1, 3]
        test_neighbours = [{0, 1}, {0, 1, 2}, {2, 3}]

        index_mapping = NeighbourIndicesMapping(len(LINEAR_MESH), periodic=False)
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )

    def test_periodic_boundaries(self):
        test_dof_indices = [0, 1, 3]
        test_neighbours = [{3, 0, 1}, {0, 1, 2}, {2, 3, 0}]

        index_mapping = NeighbourIndicesMapping(len(LINEAR_MESH), periodic=True)
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )
