from unittest import TestCase

import numpy as np

from pde_solver.discrete_solution import DiscreteSolution


class TestDiscreteSolution(TestCase):
    initial_data = np.array([0, 0])
    solution = DiscreteSolution(0, initial_data)
    solution.add_solution(
        1,
        np.array(
            [
                1,
                1,
            ]
        ),
    )

    def time(self):
        self.assertListEqual(self.solution.time, [0, 1])

    def test_solution(self):
        for i, solution_i in enumerate(self.solution.solution):
            self.assertListEqual(list(solution_i), [i, i])

    def test_dimension(self):
        self.assertEqual(self.solution.dimension, 2)

    def test_end_solution(self):
        self.assertListEqual(list(self.solution.end_solution), [1, 1])

    def test_end_time(self):
        self.assertEqual(self.solution.end_time, 1)
