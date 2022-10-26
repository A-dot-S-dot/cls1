from unittest import TestCase

import numpy as np

from pde_solver.discretization import (
    CoarseSolution,
    DiscreteSolution,
    TemporalInterpolation,
)


class TestDiscreteSolution(TestCase):
    initial_data = np.array([[0, 0], [0, 0], [0, 0]])
    solution = DiscreteSolution(0, initial_data, np.array([0, 0.5, 1]))
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


class TestTemporalInterpolation(TestCase):
    def test_interpolator(self):
        initial_data = np.array([[0, 0], [0, 0], [0, 0]])
        solution = DiscreteSolution(0, initial_data, np.array([0, 0.5, 1]))
        solution.add_solution(
            1,
            np.array([[1, 1], [1, 1], [1, 1]]),
        )
        interpolation_times = np.array([0.5])

        interpolator = TemporalInterpolation()
        interpolated_values = interpolator(solution, interpolation_times)

        self.assertTupleEqual(interpolated_values.shape, (1, 3, 2))
        self.assertListEqual(list(interpolated_values.flatten()), 6 * [0.5])


class TestCoarsenedDiscreteSolution(TestCase):
    def test_coarsening_grid(self):
        discrete_solution = DiscreteSolution(
            0, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])
        )
        discrete_solution.add_solution(1, np.array([2, 4, 6, 8]))
        expected_coarse_grid = np.array([1.5, 3.5])

        coarse_solution = CoarseSolution(discrete_solution, 2)

        for i in range(2):
            self.assertAlmostEqual(coarse_solution.grid[i], expected_coarse_grid[i])

    def test_coarsening_two_dimensions(self):
        discrete_solution = DiscreteSolution(
            0, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])
        )
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
            0, np.array([[1, 2, 3, 4], [0, 1, 2, 3]]).T, np.array([1, 2, 3, 4])
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
        discrete_solution = DiscreteSolution(
            0, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])
        )
        self.assertRaises(ValueError, CoarseSolution, discrete_solution, 3)
