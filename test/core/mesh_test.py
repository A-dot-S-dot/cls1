from test.test_helper import discrete_derivative
from unittest import TestCase

import numpy as np

from core.mesh import *


class TestInterval(TestCase):
    interval = Interval(0, 1)

    def test_type_errors(self):
        self.assertRaises(ValueError, Interval, "a", "b")
        self.assertRaises(ValueError, Interval, 1, "b")
        self.assertRaises(ValueError, Interval, "a", 2)

    def test_wrong_order_error(self):
        self.assertRaises(AssertionError, Interval, 1, 0)

    def test_equal_error(self):
        self.assertRaises(AssertionError, Interval, 0, 0)

    def test_a_getter(self):
        self.assertEqual(self.interval.a, 0)

    def test_b_getter(self):
        self.assertEqual(self.interval.b, 1)

    def test_length(self):
        self.assertAlmostEqual(self.interval.length, 1.0, 12)

    def test_is_equal(self):
        other_interval = Interval(0, 1)
        other_interval_2 = Interval(-1, 1)

        self.assertTrue(self.interval == other_interval)
        self.assertTrue(self.interval != other_interval_2)

    def test_contains(self):
        inner_points = [0.1, 0.5, 0.75, 0.99999]
        boundary_points = [0, 1]
        outer_points = [-2, 1000, 1.00000001]

        for point in inner_points:
            self.assertTrue(point in self.interval)

        for point in boundary_points:
            self.assertTrue(point in self.interval)

        for point in outer_points:
            self.assertFalse(point in self.interval)

    def test_is_in_inner(self):
        inner_points = [0.1, 0.5, 0.75, 0.99999]
        boundary_points = [0, 1]
        outer_points = [-2, 1000, 1.00000001]

        for point in inner_points:
            self.assertTrue(self.interval.is_in_inner(point))

        for point in boundary_points:
            self.assertFalse(self.interval.is_in_inner(point))

        for point in outer_points:
            self.assertFalse(self.interval.is_in_inner(point))

    def test_is_in_boundary(self):
        inner_points = [0.1, 0.5, 0.75, 0.99999]
        boundary_points = [0, 1]
        outer_points = [-2, 1000, 1.00000001]

        for point in inner_points:
            self.assertFalse(self.interval.is_in_boundary(point))

        for point in boundary_points:
            self.assertTrue(self.interval.is_in_boundary(point))

        for point in outer_points:
            self.assertFalse(self.interval.is_in_boundary(point))


class TestUniformMesh(TestCase):
    domain = Interval(0, 1)
    mesh_size = 2
    expected_cells = [Interval(0, 0.5), Interval(0.5, 1)]
    expected_space_steps = [0.5, 0.5]
    mesh = UniformMesh(domain, mesh_size)
    points = [0, 0.25, 0.5, 0.75, 1]
    expected_indices = [[0], [0], [0, 1], [1], [1]]
    expected_simplices = [
        [expected_cells[0]],
        [expected_cells[0]],
        [expected_cells[0], expected_cells[1]],
        [expected_cells[1]],
        [expected_cells[1]],
    ]

    def test_domain(self):
        self.assertEqual(self.mesh.domain, self.domain)

    def test_step_length(self):
        self.assertAlmostEqual(
            self.mesh.step_length, self.domain.length / self.mesh_size, 12
        )

    def test_space_steps(self):
        space_steps = self.mesh.space_steps
        for i in range(self.mesh_size):
            self.assertAlmostEqual(
                space_steps[i], self.expected_space_steps[i], msg=f"index={i}"
            )

    def test_length(self):
        self.assertEqual(len(self.mesh), self.mesh_size)

    def test_iter(self):
        for simplex, test_simplex in zip(self.mesh, self.expected_cells):
            self.assertEqual(simplex, test_simplex)

    def test_getitem(self):
        for index, test_cell in enumerate(self.expected_cells):
            self.assertEqual(self.mesh[index], test_cell)

    def test_eq(self):
        test_mesh = UniformMesh(self.domain, self.mesh_size)

        self.assertTrue(self.mesh == test_mesh)

    def test_not_eq(self):
        test_mesh_1 = UniformMesh(self.domain, int(round(self.mesh_size / 2)))
        test_mesh_2 = UniformMesh(Interval(0, 2), self.mesh_size)

        for test_mesh in [test_mesh_1, test_mesh_2]:
            self.assertTrue(self.mesh != test_mesh)

    def test_refine(self):
        refined_mesh = UniformMesh(self.domain, 3 * self.mesh_size)
        self.assertEqual(self.mesh.refine(3), refined_mesh)

    def test_find_cells(self):
        for point, simplices in zip(self.points, self.expected_simplices):
            self.assertListEqual(list(self.mesh.find_cells(point)), simplices)

    def test_find_cell_indices(self):
        for point, indices in zip(self.points, self.expected_indices):
            self.assertListEqual(list(self.mesh.find_cell_indices(point)), indices)

    def test_coarsen(self):
        coarsened_mesh = UniformMesh(self.domain, 1)
        self.assertEqual(self.mesh.coarsen(2), coarsened_mesh)

    def test_coarsen_error(self):
        self.assertRaises(AssertionError, self.mesh.coarsen, 3)


class TestAffineTransformation(TestCase):
    simplex = Interval(-1, 1)
    simplex_points = [-1, 0, 1]
    standard_simplex_points = [0, 0.5, 1]
    affine_transformation = AffineTransformation()

    def test_call(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(self.affine_transformation(x_standard, self.simplex), x)

    def test_inverse(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(
                self.affine_transformation.inverse(x, self.simplex), x_standard
            )

    def test_inverse_property(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(
                self.affine_transformation.inverse(
                    self.affine_transformation(x, self.simplex), self.simplex
                ),
                x,
            )
            self.assertEqual(
                self.affine_transformation(
                    self.affine_transformation.inverse(x_standard, self.simplex),
                    self.simplex,
                ),
                x_standard,
            )

    def test_derivative(self):
        for x in np.linspace(self.simplex.a, self.simplex.b):
            self.assertAlmostEqual(
                self.affine_transformation.derivative(self.simplex),
                discrete_derivative(
                    lambda x: self.affine_transformation(x, self.simplex), x
                ),
            )
