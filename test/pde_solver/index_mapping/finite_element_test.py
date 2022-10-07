from unittest import TestCase

from pde_solver.index_mapping.finite_element import (
    GlobalIndexMapping,
    DOFNeighbourIndicesMapping,
)
from pde_solver.mesh import Interval
from pde_solver.mesh.uniform import UniformMesh
from test.test_helper import LINEAR_MESH, QUADRATIC_MESH


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

    def test_boundary_condition(self):
        for polynomial_degree in range(1, 7):
            index_mapping = GlobalIndexMapping(self.mesh, polynomial_degree)
            for input in [(0, 0), (2, polynomial_degree)]:
                self.assertEqual(index_mapping(*input), 0, msg=f"input={input}")


class TestDOFNeighbourIndicesMapping(TestCase):
    def test_linear_element_space(self):
        test_dof_indices = [0]
        test_neighbours = [{0, 1, 3}]
        index_mapping = DOFNeighbourIndicesMapping(LINEAR_MESH, 1, 4)
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
        index_mapping = DOFNeighbourIndicesMapping(QUADRATIC_MESH, 2, 4)
        for dof_index, expected_neighbours in zip(test_dof_indices, test_neighbours):
            neighbours = set(index_mapping(dof_index))
            self.assertSetEqual(
                neighbours,
                expected_neighbours,
                msg=f"index={dof_index}, neighbours={neighbours}, expected_neighbours={expected_neighbours}",
            )
