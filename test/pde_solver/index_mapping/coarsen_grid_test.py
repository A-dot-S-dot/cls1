from unittest import TestCase

from pde_solver.index_mapping import FineGridIndicesMapping


class TestFineGridIndicesMapping(TestCase):
    def test_mapping(self):
        index_mapping = FineGridIndicesMapping(6, 2)
        test_coarse_cell_indices = [0, 1, 2]
        expected_fine_cell_indices = [(0, 1), (2, 3), (4, 5)]

        for edge_index, expected_indices in zip(
            test_coarse_cell_indices, expected_fine_cell_indices
        ):
            self.assertTupleEqual(index_mapping(edge_index), expected_indices)

    def test_not_admissible_error(self):
        self.assertRaises(ValueError, FineGridIndicesMapping, 6, 4)
