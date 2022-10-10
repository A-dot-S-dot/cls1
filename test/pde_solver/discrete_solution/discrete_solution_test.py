from unittest import TestCase

import numpy as np

from pde_solver.discrete_solution import DiscreteSolution


class TestDiscreteSolution(TestCase):
    initial_data = np.array([[0, 0], [0, 0], [0, 0]])
    solution = DiscreteSolution(0, initial_data)
    solution.add_solution(
        1,
        np.array([[1, 1], [1, 1], [1, 1]]),
    )

    def time(self):
        self.assertListEqual(self.solution.time, [0, 1])

    def test_solution(self):
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(self.solution.solution[i, j, k], i)

    def test_dimension(self):
        self.assertEqual(self.solution.dimension, 3)

    def test_end_solution(self):
        for j in range(3):
            for k in range(2):
                self.assertEqual(self.solution.end_solution[j, k], 1)

    def test_end_time(self):
        self.assertEqual(self.solution.end_time, 1)
