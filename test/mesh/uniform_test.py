from unittest import TestCase

from mesh import Interval
from mesh.uniform import UniformMesh


class TestUniformMesh(TestCase):
    interval = Interval(0, 1)
    element_number = 2
    test_simplices = [Interval(0, 0.5), Interval(0.5, 1)]
    mesh = UniformMesh(interval, element_number)
    points = [0, 0.25, 0.5, 0.75, 1]
    expected_indices = [[0], [0], [0, 1], [1], [1]]
    expected_simplices = [
        [test_simplices[0]],
        [test_simplices[0]],
        [test_simplices[0], test_simplices[1]],
        [test_simplices[1]],
        [test_simplices[1]],
    ]

    def test_nodes_number_not_positive_error(self):
        self.assertRaises(ValueError, UniformMesh, self.interval, 0)
        self.assertRaises(ValueError, UniformMesh, self.interval, -2)

    def test_step_length(self):
        self.assertAlmostEqual(
            self.mesh.step_length, self.interval.length / self.element_number, 12
        )

    def test_domain(self):
        self.assertEqual(self.mesh.domain, self.interval)

    def test_length(self):
        self.assertEqual(len(self.mesh), self.element_number)

    def test_iter(self):
        for simplex, test_simplex in zip(self.mesh, self.test_simplices):
            self.assertEqual(simplex, test_simplex)

    def test_getitem(self):
        for index, test_simplex in zip(
            range(len(self.test_simplices)), self.test_simplices
        ):
            self.assertEqual(self.mesh[index], test_simplex)

    def test_eq(self):
        test_mesh = UniformMesh(self.interval, self.element_number)

        self.assertTrue(self.mesh == test_mesh)

    def test_not_eq(self):
        test_mesh_1 = UniformMesh(self.interval, int(round(self.element_number / 2)))
        test_mesh_2 = UniformMesh(Interval(0, 2), self.element_number)

        for test_mesh in [test_mesh_1, test_mesh_2]:
            self.assertTrue(self.mesh != test_mesh)

    def test_index(self):
        for index, test_simplex in zip(
            range(len(self.test_simplices)), self.test_simplices
        ):
            self.assertEqual(self.mesh.index(test_simplex), index)

    def test_find_simplices(self):
        for point, simplices in zip(self.points, self.expected_simplices):
            self.assertListEqual(self.mesh.find_simplices(point), simplices)

    def test_find_simplex_indices(self):
        for point, indices in zip(self.points, self.expected_indices):
            self.assertListEqual(self.mesh.find_simplex_indices(point), indices)

    def test_refine(self):
        refined_mesh = UniformMesh(self.interval, 2 * self.element_number)
        self.assertEqual(self.mesh.refine(), refined_mesh)
