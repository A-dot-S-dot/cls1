from itertools import product
from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import core

from command.error import *


class TestErrorEvolutionCalculator(TestCase):
    calculator = ErrorEvolutionCalculator()

    def test_different_spatial_dimension_error(self):
        vector = np.array([[0.25, 0.25], [0.75, 0.75]])
        solution = core.DiscreteSolutionWithHistory(vector + 0.1)
        solution.update(1.0, vector + 0.9)

        vector_exact = np.array(
            [[1 / 8, 1 / 8], [3 / 8, 3 / 8], [5 / 8, 5 / 8], [7 / 8, 7 / 8]]
        )
        solution_exact = core.DiscreteSolutionWithHistory(
            vector_exact, space=VOLUME_SPACE
        )
        solution_exact.update(0.5, vector_exact + 0.5)
        solution_exact.update(0.5, vector_exact + 0.5)

        self.assertRaises(ValueError, self.calculator, solution, solution_exact)

    def test_no_norm_can_be_created_error(self):
        vector = np.array([[0.25, 0.25], [0.75, 0.75]])
        solution = core.DiscreteSolutionWithHistory(vector + 0.1)
        solution.update(1.0, vector + 0.9)

        solution_exact = core.DiscreteSolutionWithHistory(vector)
        solution_exact.update(0.5, vector + 0.5)
        solution_exact.update(0.5, vector + 0.5)

        self.assertRaises(ValueError, self.calculator, solution, solution_exact)

    def test_error_evolution_calculator(self):
        vector = np.array(
            [[1 / 8, 1 / 8], [3 / 8, 3 / 8], [5 / 8, 5 / 8], [7 / 8, 7 / 8]]
        )
        solution = core.DiscreteSolutionWithHistory(vector + 0.1)
        solution.update(1.0, vector + 1.1)

        solution_exact = core.DiscreteSolutionWithHistory(vector, space=VOLUME_SPACE)
        solution_exact.update(0.5, vector + 0.5)
        solution_exact.update(0.5, vector + 1.0)

        time, error = self.calculator(solution, solution_exact)

        expected_time = np.array([0.0, 0.5, 1.0])
        expected_error = 0.1 * np.ones((3, 2))

        self.assertTrue(np.array_equal(time, expected_time))

        for i, j in product(*[range(dim) for dim in error.shape]):
            self.assertAlmostEqual(error[i, j], expected_error[i, j])
