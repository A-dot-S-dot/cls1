from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import core
import numpy as np
from finite_volume import FiniteVolumeSpace
from numpy.testing import assert_almost_equal, assert_equal

from command.error_evolution import *


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

        self.assertRaises(AssertionError, self.calculator, solution, solution_exact)

    def test_no_norm_can_be_created_error(self):
        vector = np.array([[0.25, 0.25], [0.75, 0.75]])
        solution = core.DiscreteSolutionWithHistory(vector + 0.1)
        solution.update(1.0, vector + 0.9)

        solution_exact = core.DiscreteSolutionWithHistory(vector)
        solution_exact.update(0.5, vector + 0.5)
        solution_exact.update(0.5, vector + 0.5)

        self.assertRaises(ValueError, self.calculator, solution, solution_exact)

    def test_error_evolution_calculator(self):
        vector = np.array([[1.0, 1.0], [1.0, 1.0]])
        vector2 = np.array([[1 / 2, 1 / 2], [3 / 2, 3 / 2]])
        vector3 = np.array([[0.0, 0.0], [2.0, 2.0]])
        space = FiniteVolumeSpace(core.UniformMesh(core.Interval(0, 1), 2))
        solution_exact = core.DiscreteSolutionWithHistory(vector, space=space)
        solution_exact.update(0.5, vector2)
        solution_exact.update(0.5, vector3)

        solution = core.DiscreteSolutionWithHistory(vector + 0.1)
        solution.update(1.0, vector3 + 0.1)

        time, error = self.calculator(
            solution, solution_exact, norm=core.L1Norm(space.mesh)
        )

        expected_time = np.array([0.0, 0.5, 1.0])
        expected_error = 0.1 * np.ones((3, 2))

        assert_equal(time, expected_time)
        assert_almost_equal(error, expected_error)
