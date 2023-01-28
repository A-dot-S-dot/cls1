from itertools import product
from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import core

from command.error import *


class TestErrorEvolutionCalculator(TestCase):
    def test_error_evolution_calculator(self):
        vector = np.array(
            [[1 / 8, 1 / 8], [3 / 8, 3 / 8], [5 / 8, 5 / 8], [7 / 8, 7 / 8]]
        )
        solution = core.DiscreteSolutionWithHistory(vector + 0.1)
        solution_exact = core.DiscreteSolutionWithHistory(vector)

        solution.update(1.0, vector + 0.9)
        solution_exact.update(1.0, vector + 1)

        calculator = ErrorEvolutionCalculator(VOLUME_SPACE)
        time, error = calculator(solution, solution_exact)

        expected_time = np.array([0.0, 1.0])
        expected_error = 0.1 * np.ones((2, 2))

        self.assertTrue((time == expected_time).all())

        for i, j in product([0, 1], repeat=2):
            self.assertAlmostEqual(error[i, j], expected_error[i, j])
