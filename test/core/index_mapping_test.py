from test.test_helper import LINEAR_MESH, QUADRATIC_MESH
from unittest import TestCase

import core.index_mapping as im
from core.mesh import Interval, UniformMesh


class TestGlobalIndexMapping(TestCase):

    interval = Interval(0, 1)
    mesh_size = 3
    mesh = UniformMesh(interval, mesh_size)

    def test_inner_point(self):
        for polynomial_degree in range(1, 7):
            index_mapping = im.GlobalIndexMapping(self.mesh, polynomial_degree)
            input = (1, 1)
            expected_output = polynomial_degree + 1

            self.assertEqual(index_mapping(*input), expected_output)

    def test_boundary_condition(self):
        for polynomial_degree in range(1, 7):
            index_mapping = im.GlobalIndexMapping(self.mesh, polynomial_degree)
            for input in [(0, 0), (2, polynomial_degree)]:
                self.assertEqual(index_mapping(*input), 0, msg=f"input={input}")


class TestDOFNeighbourIndicesMapping(TestCase):
    def test_linear_element_space(self):
        test_dof_indices = [0]
        test_neighbours = [{0, 1, 3}]
        index_mapping = im.DOFNeighbourIndicesMapping(LINEAR_MESH, 1, 4)
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )

    def test_quadratic_element_space(self):
        test_dof_indices = [0, 1]
        test_neighbours = [{0, 1, 2, 3}, {0, 1, 2}]
        index_mapping = im.DOFNeighbourIndicesMapping(QUADRATIC_MESH, 2, 4)
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )


class TestFineGridIndicesMapping(TestCase):
    def test_mapping(self):
        index_mapping = im.FineGridIndicesMapping(6, 2)
        test_coarse_cell_indices = [0, 1, 2]
        expected_fine_cell_indices = [(0, 1), (2, 3), (4, 5)]

        for cell_index, expected_indices in zip(
            test_coarse_cell_indices, expected_fine_cell_indices
        ):
            self.assertTupleEqual(index_mapping(cell_index), expected_indices)

    def test_not_admissible_error(self):
        self.assertRaises(ValueError, im.FineGridIndicesMapping, 6, 4)


class TestCellIndexMapping(TestCase):
    def test_mapping(self):
        index_mapping = im.LeftRightCellIndexMapping(LINEAR_MESH)
        test_node_indices = [0, 1, 2, 3]
        expected_cell_indices = [(3, 0), (0, 1), (1, 2), (2, 3)]

        for node_index, expected_indices in zip(
            test_node_indices, expected_cell_indices
        ):
            self.assertTupleEqual(index_mapping(node_index), expected_indices)


class TestNodeIndexMapping(TestCase):
    def test_mapping(self):
        index_mapping = im.LeftRightNodeIndexMapping(LINEAR_MESH)
        test_cell_indices = [0, 1, 2, 3]
        expected_node_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for cell_index, expected_indices in zip(
            test_cell_indices, expected_node_indices
        ):
            self.assertTupleEqual(index_mapping(cell_index), expected_indices)
