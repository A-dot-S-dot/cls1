from test.test_helper import LINEAR_MESH, QUADRATIC_MESH
from unittest import TestCase

from core.finite_element.index_mapping import *
from core.mesh import Interval, UniformMesh


class TestGlobalIndexMapping(TestCase):

    interval = Interval(0, 1)
    mesh_size = 3
    mesh = UniformMesh(interval, mesh_size)

    def test_inner_point(self):
        for polynomial_degree in range(1, 7):
            index_mapping = GlobalIndexMapping(self.mesh, polynomial_degree)
            input = (1, 1)
            expected_output = polynomial_degree + 1

            self.assertEqual(index_mapping(*input), expected_output)

    def test_periodic_boundary_condition(self):
        for polynomial_degree in range(1, 7):
            index_mapping = GlobalIndexMapping(
                self.mesh, polynomial_degree, periodic=True
            )
            for input in [(0, 0), (2, polynomial_degree)]:
                self.assertEqual(index_mapping(*input), 0, msg=f"input={input}")

    def test_non_periodic_boundary_condition(self):
        for polynomial_degree in range(1, 7):
            index_mapping = GlobalIndexMapping(
                self.mesh, polynomial_degree, periodic=False
            )
            self.assertEqual(index_mapping(0, 0), 0, msg=f"input={input}")
            self.assertEqual(index_mapping(2, polynomial_degree), 3 * polynomial_degree)


class TestDOFNeighbourIndicesMapping(TestCase):
    def test_linear_element_space_with_periodic_boundary(self):
        test_dof_indices = [0]
        test_neighbours = [{0, 1, 3}]
        index_mapping = DOFNeighbourIndicesMapping(
            GlobalIndexMapping(LINEAR_MESH, 1, periodic=True)
        )
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )

    def test_linear_element_space_without_periodic_boundary(self):
        test_dof_indices = [0]
        test_neighbours = [{0, 1}]
        index_mapping = DOFNeighbourIndicesMapping(
            GlobalIndexMapping(LINEAR_MESH, 1, periodic=False)
        )
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )

    def test_quadratic_element_space_with_periodic_boundary(self):
        test_dof_indices = [0, 1]
        test_neighbours = [{0, 1, 2, 3}, {0, 1, 2}]
        index_mapping = DOFNeighbourIndicesMapping(
            GlobalIndexMapping(QUADRATIC_MESH, 2, periodic=True)
        )
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )

    def test_quadratic_element_space_without_periodic_boundary(self):
        test_dof_indices = [0, 1, 2, 4]
        test_neighbours = [{0, 1, 2}, {0, 1, 2}, {0, 1, 2, 3, 4}, {2, 3, 4}]
        index_mapping = DOFNeighbourIndicesMapping(
            GlobalIndexMapping(QUADRATIC_MESH, 2, periodic=False)
        )
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )
