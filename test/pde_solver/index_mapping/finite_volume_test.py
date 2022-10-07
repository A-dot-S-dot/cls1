from test.test_helper import LINEAR_MESH
from unittest import TestCase

from pde_solver.index_mapping.finite_volume import (
    LeftRightCellIndexMapping,
    LeftRightEdgeIndexMapping,
)


class TestCellIndexMapping(TestCase):
    def test_mapping(self):
        index_mapping = LeftRightCellIndexMapping(LINEAR_MESH)
        test_edge_indices = [0, 1, 2, 3]
        expected_cell_indices = [(3, 0), (0, 1), (1, 2), (2, 3)]

        for edge_index, expected_indices in zip(
            test_edge_indices, expected_cell_indices
        ):
            self.assertTupleEqual(index_mapping(edge_index), expected_indices)


class TestEdgeIndexMapping(TestCase):
    def test_mapping(self):
        index_mapping = LeftRightEdgeIndexMapping(LINEAR_MESH)
        test_cell_indices = [0, 1, 2, 3]
        expected_edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for edge_index, expected_indices in zip(
            test_cell_indices, expected_edge_indices
        ):
            self.assertTupleEqual(index_mapping(edge_index), expected_indices)
