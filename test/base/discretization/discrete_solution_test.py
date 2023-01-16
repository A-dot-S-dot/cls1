from unittest import TestCase

import numpy as np
from base.discretization import DiscreteSolution


class TestDiscreteSolution(TestCase):
    initial_data = np.array([[0, 0], [0, 0], [0, 0]])
    solution = DiscreteSolution(initial_data, grid=np.array([0, 0.5, 1]))
    solution.add_solution(
        1,
        np.array([[1, 1], [1, 1], [1, 1]]),
    )

    def test_time(self):
        self.assertListEqual(list(self.solution.time), [0, 1])

    def test_time_steps(self):
        self.assertListEqual(self.solution.time_steps, [1])

    def test_values(self):
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(self.solution.values[i, j, k], i)

    def test_dimension(self):
        self.assertTupleEqual(self.solution.dimension, (3, 2))

    def test_end_solution(self):
        for j in range(3):
            for k in range(2):
                self.assertEqual(self.solution.end_values[j, k], 1)

    def test_end_time(self):
        self.assertEqual(self.solution.end_time, 1)
