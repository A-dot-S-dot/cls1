from unittest import TestCase

import numpy as np
from base.discretization import CoarseSolution, DiscreteSolution


class TestDiscreteSolution(TestCase):
    def create_solution(self, **kwargs) -> DiscreteSolution:
        initial_data = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        return DiscreteSolution(initial_data, **kwargs)

    def update(self, solution: DiscreteSolution):
        solution.update(
            1.0,
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        )

    def test_scalar_dimension(self):
        initial_data = np.array([0.0, 0.0])
        solution = DiscreteSolution(initial_data)
        self.assertEqual(solution.dimension, 2)

    def test_system_dimension(self):
        solution = self.create_solution()
        self.assertTupleEqual(solution.dimension, (3, 2))

    def test_dimension_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        self.assertTupleEqual(solution.dimension, (3, 2))

    def test_time_with_default_start_time_before_update(self):
        solution = self.create_solution()
        self.assertEqual(solution.time, 0.0)

    def test_time_with_default_start_time_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        self.assertEqual(solution.time, 1.0)

    def test_time_with_custom_start_time_before_update(self):
        solution = self.create_solution(start_time=1.0)
        self.assertEqual(solution.time, 1.0)

    def test_end_time_with_custom_start_time_after_update(self):
        solution = self.create_solution(start_time=1.0)
        self.update(solution)
        self.assertEqual(solution.time, 2.0)

    def test_time_history_with_default_start_time_before_update(self):
        solution = self.create_solution(save_history=True)
        self.assertListEqual(list(solution.time_history), [0.0])

    def test_time_history_with_custom_start_time_before_update(self):
        solution = self.create_solution(start_time=1.0, save_history=True)
        self.assertListEqual(list(solution.time_history), [1.0])

    def test_time_history_with_default_start_time_after_update(self):
        solution = self.create_solution(save_history=True)
        self.update(solution)
        self.assertListEqual(list(solution.time_history), [0.0, 1.0])

    def test_time_history_with_custom_start_time_after_update(self):
        solution = self.create_solution(start_time=1.0, save_history=True)
        self.update(solution)
        self.assertListEqual(list(solution.time_history), [1.0, 2.0])

    def test_no_time_history(self):
        solution = self.create_solution(save_history=False)
        self.assertRaises(AttributeError, lambda: solution.time_history)

    def test_time_step_history_befote_update(self):
        solution = self.create_solution(save_history=True)
        self.assertListEqual(list(solution.time_step_history), [])

    def test_time_step_history_after_update(self):
        solution = self.create_solution(save_history=True)
        self.update(solution)
        self.assertListEqual(list(solution.time_step_history), [1.0])

    def test_no_time_step_history(self):
        solution = self.create_solution(save_history=False)
        self.assertRaises(AttributeError, lambda: solution.time_step_history)

    def test_grid(self):
        grid = np.array([0.0, 0.5, 1.0])
        solution = self.create_solution(grid=grid)
        self.assertListEqual(list(solution.grid), list(grid))

    def test_no_grid_error(self):
        solution = self.create_solution()
        self.assertRaises(AttributeError, lambda: solution.grid)

    def test_no_space_error(self):
        solution = self.create_solution()
        self.assertRaises(AttributeError, lambda: solution.space)

    def test_value_before_update(self):
        solution = self.create_solution()
        for i in range(3):
            for j in range(2):
                self.assertEqual(solution.value[i, j], 0.0)

    def test_value_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        for i in range(3):
            for j in range(2):
                self.assertEqual(solution.value[i, j], 1.0)

    def test_value_history_before_update(self):
        solution = self.create_solution(save_history=True)
        for i in range(1):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(solution.value_history[i, j, k], i)

    def test_value_history_after_update(self):
        solution = self.create_solution(save_history=True)
        self.update(solution)
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(solution.value_history[i, j, k], i)


# class TestCoarsenedDiscreteSolution(TestCase):
#     def test_coarsening_grid(self):
#         discrete_solution = DiscreteSolution(
#             np.array([1, 2, 3, 4]), grid=np.array([1, 2, 3, 4])
#         )
#         discrete_solution.add_solution(1, np.array([2, 4, 6, 8]))
#         expected_coarse_grid = np.array([1.5, 3.5])

#         coarse_solution = CoarseSolution(discrete_solution, 2)

#         for i in range(2):
#             self.assertAlmostEqual(coarse_solution.grid[i], expected_coarse_grid[i])

#     def test_coarsening_two_dimensions(self):
#         discrete_solution = DiscreteSolution(
#             np.array([1, 2, 3, 4]), grid=np.array([1, 2, 3, 4])
#         )
#         discrete_solution.add_solution(1, np.array([2, 4, 6, 8]))
#         expected_coarsening = np.array([[1.5, 3.5], [3, 7]])

#         coarse_solution = CoarseSolution(discrete_solution, 2)

#         for i in range(len(discrete_solution.values)):
#             for j in range(2):
#                 self.assertAlmostEqual(
#                     coarse_solution.values[i, j], expected_coarsening[i, j]
#                 )

#     def test_coarsening_three_dimension(self):
#         discrete_solution = DiscreteSolution(
#             np.array([[1, 2, 3, 4], [0, 1, 2, 3]]).T, grid=np.array([1, 2, 3, 4])
#         )
#         discrete_solution.add_solution(1, np.array([[2, 4, 6, 8], [0, 2, 4, 6]]).T)
#         expected_coarsening = np.array([[[1.5, 0.5], [3.5, 2.5]], [[3, 1], [7, 5]]])
#         coarse_solution = CoarseSolution(discrete_solution, 2)

#         for i in range(len(discrete_solution.values)):
#             for j in range(2):
#                 for k in range(2):
#                     self.assertAlmostEqual(
#                         coarse_solution.values[i, j, k],
#                         expected_coarsening[i, j, k],
#                     )

#     def test_admissible_coarsening_degree(self):
#         discrete_solution = DiscreteSolution(
#             np.array([1, 2, 3, 4]), grid=np.array([1, 2, 3, 4])
#         )
#         self.assertRaises(ValueError, CoarseSolution, discrete_solution, 3)
#     def test_dimension(self):
#         self.assertTupleEqual(self.solution.dimension, (3, 2))

#     def test_end_solution(self):
#         for j in range(3):
#             for k in range(2):
#                 self.assertEqual(self.solution.end_values[j, k], 1)

#     def test_end_time(self):
#         self.assertEqual(self.solution.end_time, 1)
#     def test_no_value_history_error(self):
#         solution = self.create_solution(save_history=False)
#         self.assertRaises(AttributeError, lambda: solution.value_history)
