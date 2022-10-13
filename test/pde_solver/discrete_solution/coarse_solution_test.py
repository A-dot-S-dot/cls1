from unittest import TestCase

import numpy as np

from pde_solver.discrete_solution import CoarseSolution, DiscreteSolution


class TestCoarsenedDiscreteSolution(TestCase):
    def test_coarsening_two_dimensions(self):
        discrete_solution = DiscreteSolution(0, np.array([1, 2, 3, 4]))
        discrete_solution.add_solution(1, np.array([2, 4, 6, 8]))
        expected_coarsening = np.array([[1.5, 3.5], [3, 7]])
        coarse_solution = CoarseSolution(discrete_solution, 2)

        for i in range(len(discrete_solution.values)):
            for j in range(2):
                self.assertAlmostEqual(
                    coarse_solution.values[i, j], expected_coarsening[i, j]
                )

    def test_coarsening_three_dimension(self):
        discrete_solution = DiscreteSolution(
            0, np.array([[1, 2, 3, 4], [0, 1, 2, 3]]).T
        )
        discrete_solution.add_solution(1, np.array([[2, 4, 6, 8], [0, 2, 4, 6]]).T)
        expected_coarsening = np.array([[[1.5, 0.5], [3.5, 2.5]], [[3, 1], [7, 5]]])
        coarse_solution = CoarseSolution(discrete_solution, 2)

        for i in range(len(discrete_solution.values)):
            for j in range(2):
                for k in range(2):
                    self.assertAlmostEqual(
                        coarse_solution.values[i, j, k],
                        expected_coarsening[i, j, k],
                    )

    def test_admissible_coarsening_degree(self):
        discrete_solution = DiscreteSolution(0, np.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, CoarseSolution, discrete_solution, 3)
