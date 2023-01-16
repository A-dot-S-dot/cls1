from unittest import TestCase

from base.mesh import Interval, UniformMesh


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
