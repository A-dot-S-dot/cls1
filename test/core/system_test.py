from test.test_helper import LINEAR_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np

from core import system


class SimpleSystemMatrix(system.SystemMatrix):
    def __init__(self):
        system.SystemMatrix.__init__(self, 4)
        self.assemble()

    def assemble(self):
        for i in range(self.dimension):
            self[i, i] = i + 1

        self.update_csr_values()


class TestSystemMatrix(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE

    def test_dimension(self):
        matrix = SimpleSystemMatrix()
        self.assertEqual(matrix.dimension, self.element_space.dimension)

    def test_assemble(self):
        matrix = SimpleSystemMatrix()
        for i in range(matrix.dimension):
            self.assertEqual(matrix[i, i], i + 1)

    def test_assemble_with_array(self):
        matrix = SimpleSystemMatrix()
        matrix.set_values(np.ones((matrix.dimension, matrix.dimension)))
        for i in range(matrix.dimension):
            for j in range(matrix.dimension):
                self.assertEqual(matrix[i, j], 1)

    def test_inverse(self):
        matrix = SimpleSystemMatrix()
        b = np.array([1, 2, 0, 0])
        x = matrix.inverse(b)
        expected_x = np.array([1, 1, 0, 0])

        self.assertListEqual(list(x), list(expected_x))

    def test_permanent_inverse(self):
        matrix = SimpleSystemMatrix()
        matrix.build_inverse()

        b = np.array([1, 2, 0, 0])
        x = matrix.inverse(b)
        expected_x = np.array([1, 1, 0, 0])

        self.assertListEqual(list(x), list(expected_x))

    def test_addition(self):
        matrix1 = SimpleSystemMatrix()
        matrix2 = SimpleSystemMatrix()

        matrix = matrix1 + matrix2
        expected_sum = np.diag([2, 4, 6, 8])

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(matrix[i, j], expected_sum[i, j])

    def test_subtraction(self):
        matrix1 = SimpleSystemMatrix()
        matrix2 = SimpleSystemMatrix()

        matrix = matrix1 - matrix2
        expected_sum = np.zeros(
            (self.element_space.dimension, self.element_space.dimension)
        )

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(matrix[i, j], expected_sum[i, j])

    def test_multiply_row(self):
        matrix = SimpleSystemMatrix()
        matrix.set_values(np.arange(16).reshape((4, 4)))

        dof_vector = np.array([1, 2, 3, 4])

        new_matrix = matrix.multiply_row(dof_vector).tolil()

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(new_matrix[i, j], matrix[i, j] * dof_vector[j])

    def test_multiply_column(self):
        matrix = SimpleSystemMatrix()
        matrix.set_values(np.arange(16).reshape((4, 4)))

        dof_vector = np.array([1, 2, 3, 4])

        new_matrix = matrix.multiply_column(dof_vector).tolil()

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(new_matrix[i, j], matrix[i, j] * dof_vector[i])
