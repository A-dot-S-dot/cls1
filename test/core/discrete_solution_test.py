from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from core.discrete_solution import *


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

    def test_time_with_default_initial_time_before_update(self):
        solution = self.create_solution()
        self.assertEqual(solution.time, 0.0)

    def test_time_with_default_initial_time_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        self.assertEqual(solution.time, 1.0)

    def test_time_with_custom_initial_time_before_update(self):
        solution = self.create_solution(initial_time=1.0)
        self.assertEqual(solution.time, 1.0)

    def test_end_time_with_custom_initial_time_after_update(self):
        solution = self.create_solution(initial_time=1.0)
        self.update(solution)
        self.assertEqual(solution.time, 2.0)

    def test_grid(self):
        solution = self.create_solution(space=VOLUME_SPACE)
        assert_equal(solution.grid, VOLUME_SPACE.grid)

    def test_no_grid_error(self):
        solution = self.create_solution(space=None)
        self.assertRaises(AttributeError, lambda: solution.grid)

    def test_value_before_update(self):
        solution = self.create_solution()
        assert_equal(solution.value, 0.0)

    def test_value_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        assert_equal(solution.value, 1.0)

    def test_not_finite_error(self):
        solution = self.create_solution()
        self.assertRaises(
            ValueError,
            solution.update,
            1.0,
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, np.nan]]),
        )


class TestDiscreteSolutionWithHistory(TestCase):
    def create_solution(self, **kwargs) -> DiscreteSolutionWithHistory:
        initial_data = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        solution = DiscreteSolution(initial_data, **kwargs)
        return DiscreteSolutionWithHistory(solution)

    def update(self, solution: DiscreteSolution):
        solution.update(
            1.0,
            np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        )

    def test_time_history_with_default_initial_time_before_update(self):
        solution = self.create_solution()
        assert_equal(solution.time_history, 0.0)

    def test_time_history_with_custom_initial_time_before_update(self):
        solution = self.create_solution(initial_time=1.0)
        assert_equal(solution.time_history, 1.0)

    def test_time_history_with_default_initial_time_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        assert_equal(solution.time_history, [0.0, 1.0])

    def test_time_history_with_custom_initial_time_after_update(self):
        solution = self.create_solution(initial_time=1.0)
        self.update(solution)
        assert_equal(solution.time_history, [1.0, 2.0])

    def test_time_step_history_befote_update(self):
        solution = self.create_solution()
        assert_equal(solution.time_step_history, [])

    def test_time_step_history_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        assert_equal(solution.time_step_history, 1.0)

    def test_value_history_before_update(self):
        solution = self.create_solution()
        for i in range(1):
            assert_equal(solution.value_history[i], i)

    def test_value_history_after_update(self):
        solution = self.create_solution()
        self.update(solution)
        for i in range(2):
            assert_equal(solution.value_history[i], i)

    def test_set_value(self):
        solution = self.create_solution()
        self.update(solution)
        self.update(solution)
        value = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        solution.set_value(value)

        assert_equal(solution.value_history, value[None, :])
        assert_equal(solution.time_history, 0.0)


class TestCoarseSolution(TestCase):
    def test_coarse_solution(self):
        solution = DiscreteSolution(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        coarse_solution = CoarseSolution(solution, coarsening_degree=2)
        expected_value = np.array([[2.0, 3.0], [6.0, 7.0]])
        assert_equal(coarse_solution.value, expected_value)


class TestCoarseSolutionWithHistory(TestCase):
    def test_coarse_solution(self):
        solution = DiscreteSolution(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        )
        coarse_solution = CoarseSolutionWithHistory(
            solution,
            coarsening_degree=2,
        )
        coarse_solution.update(
            1.0, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        expected_value_history = np.array(
            [[[2.0, 3.0], [6.0, 7.0]], [[2.0, 3.0], [6.0, 7.0]]]
        )

        assert_equal(coarse_solution.value_history, expected_value_history)
