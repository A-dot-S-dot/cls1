from unittest import TestCase

from pde_solver.solver_space import LagrangeFiniteElementSpace
from pde_solver.mesh import Interval, UniformMesh


class TestLinearLagrangeFiniteElementSpace(TestCase):
    polynomial_degree = 1
    mesh_size = 2
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, mesh_size)
    element_space = LagrangeFiniteElementSpace(mesh, polynomial_degree)
    expected_basis_nodes = [0, 0.5]

    def test_basis_nodes(self):
        self.assertListEqual(
            list(self.element_space.basis_nodes), self.expected_basis_nodes
        )


class TestQuadraticLagrangeFiniteElementSpace(TestLinearLagrangeFiniteElementSpace):
    polynomial_degree = 2
    mesh_size = 2
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, mesh_size)
    element_space = LagrangeFiniteElementSpace(mesh, polynomial_degree)
    expected_basis_nodes = [0, 0.25, 0.5, 0.75]
