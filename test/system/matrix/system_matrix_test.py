from unittest import TestCase

import numpy as np
from fem import FiniteElementSpace
from system.matrix import SystemMatrix, SystemMatrixEntryCalculator
from system.matrix.system_matrix import LocallyAssembledSystemMatrix
from system.vector.dof_vector import DOFVector

from ...test_helper import LINEAR_LAGRANGE_SPACE


class SimpleSystemMatrix(SystemMatrix):
    def __init__(self, element_space: FiniteElementSpace):
        SystemMatrix.__init__(self, element_space)
        self.assemble()

    def assemble(self):
        for i in range(self.dimension):
            self[i, i] = i + 1

        super().assemble()


class SimpleMatrixEntryCalculator(SystemMatrixEntryCalculator):
    def __call__(
        self, simplex_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        return 0


class TestSystemMatrix(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE

    def test_dimension(self):
        matrix = SimpleSystemMatrix(self.element_space)
        self.assertEqual(matrix.dimension, self.element_space.dimension)

    def test_assemble(self):
        matrix = SimpleSystemMatrix(self.element_space)
        for i in range(matrix.dimension):
            self.assertEqual(matrix[i, i], i + 1)

    def test_assemble_with_array(self):
        matrix = SimpleSystemMatrix(self.element_space)
        matrix.set_values(np.ones((matrix.dimension, matrix.dimension)))
        for i in range(matrix.dimension):
            for j in range(matrix.dimension):
                self.assertEqual(matrix[i, j], 1)

    def test_inverse(self):
        matrix = SimpleSystemMatrix(self.element_space)
        b = np.array([1, 2, 0, 0])
        x = matrix.inverse(b)
        expected_x = np.array([1, 1, 0, 0])

        self.assertListEqual(list(x), list(expected_x))

    def test_permanent_inverse(self):
        matrix = SimpleSystemMatrix(self.element_space)
        matrix.build_inverse()

        b = np.array([1, 2, 0, 0])
        x = matrix.inverse(b)
        expected_x = np.array([1, 1, 0, 0])

        self.assertListEqual(list(x), list(expected_x))

    def test_addition(self):
        matrix1 = SimpleSystemMatrix(self.element_space)
        matrix2 = SimpleSystemMatrix(self.element_space)

        matrix = matrix1 + matrix2
        expected_sum = np.diag([2, 4, 6, 8])

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(matrix[i, j], expected_sum[i, j])

    def test_subtraction(self):
        matrix1 = SimpleSystemMatrix(self.element_space)
        matrix2 = SimpleSystemMatrix(self.element_space)

        matrix = matrix1 - matrix2
        expected_sum = np.zeros(
            (self.element_space.dimension, self.element_space.dimension)
        )

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(matrix[i, j], expected_sum[i, j])

    def test_multiply_row(self):
        matrix = SimpleSystemMatrix(self.element_space)
        matrix.set_values(np.arange(16).reshape((4, 4)))

        dof_vector = DOFVector(self.element_space)
        dof_vector.dofs = np.array([1, 2, 3, 4])

        new_matrix = matrix.multiply_row(dof_vector).tolil()

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(new_matrix[i, j], matrix[i, j] * dof_vector[j])

    def test_multiply_column(self):
        matrix = SimpleSystemMatrix(self.element_space)
        matrix.set_values(np.arange(16).reshape((4, 4)))
        print(matrix)

        dof_vector = DOFVector(self.element_space)
        dof_vector.dofs = np.array([1, 2, 3, 4])

        new_matrix = matrix.multiply_column(dof_vector).tolil()
        print(new_matrix.toarray())

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(new_matrix[i, j], matrix[i, j] * dof_vector[i])


class TestLocallyAssembledSystemMatrix(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    entry_calculator = SimpleMatrixEntryCalculator()

    def test_assemble(self):
        expected_sum = np.zeros(
            (self.element_space.dimension, self.element_space.dimension)
        )
        matrix = LocallyAssembledSystemMatrix(self.element_space, self.entry_calculator)
        matrix.assemble()

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(matrix[i, j], 0)
