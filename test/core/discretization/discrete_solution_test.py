from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
from core.discretization import DiscreteSolution, DiscreteSolutionWithHistory


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

    def test_grid(self):
        solution = self.create_solution(space=VOLUME_SPACE)
        self.assertListEqual(list(solution.grid), list(VOLUME_SPACE.grid))

    def test_no_grid_error(self):
        solution = self.create_solution(space=None)
        self.assertRaises(AttributeError, lambda: solution.grid)

    def test_no_space_error(self):
        solution = self.create_solution(space=None)
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


class TestDiscreteSolutionWithHistory(TestCase):
    def create_solution(self, **kwargs) -> DiscreteSolutionWithHistory:
        initial_data = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        return DiscreteSolutionWithHistory(initial_data, **kwargs)

    def update(self, solution: DiscreteSolution):
        solution.update(
            1.0,
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        )

    def test_time_history_with_default_start_time_before_update(self):
        solution = self.create_solution()
        self.assertListEqual(list(solution.time_history), [0.0])

    def test_time_history_with_custom_start_time_before_update(self):
        solution = self.create_solution(start_time=1.0)
        self.assertListEqual(list(solution.time_history), [1.0])

    def test_time_history_with_default_start_time_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        self.assertListEqual(list(solution.time_history), [0.0, 1.0])

    def test_time_history_with_custom_start_time_after_update(self):
        solution = self.create_solution(start_time=1.0)
        self.update(solution)
        self.assertListEqual(list(solution.time_history), [1.0, 2.0])

    def test_time_step_history_befote_update(self):
        solution = self.create_solution()
        self.assertListEqual(list(solution.time_step_history), [])

    def test_time_step_history_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        self.assertListEqual(list(solution.time_step_history), [1.0])

    def test_value_history_before_update(self):
        solution = self.create_solution()
        for i in range(1):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(solution.value_history[i, j, k], i)

    def test_value_history_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(solution.value_history[i, j, k], i)
