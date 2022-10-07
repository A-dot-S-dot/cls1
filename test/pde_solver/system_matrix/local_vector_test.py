from test.test_helper import LINEAR_LAGRANGE_SPACE
from unittest import TestCase

from pde_solver.system_matrix import SystemMatrixEntryCalculator
from pde_solver.system_matrix.local_matrix import LocallyAssembledSystemMatrix


class SimpleMatrixEntryCalculator(SystemMatrixEntryCalculator):
    def __call__(
        self, cell_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        return 0


class TestLocallyAssembledSystemMatrix(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    entry_calculator = SimpleMatrixEntryCalculator()

    def test_assemble(self):
        matrix = LocallyAssembledSystemMatrix(self.element_space, self.entry_calculator)

        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertEqual(matrix[i, j], 0)
