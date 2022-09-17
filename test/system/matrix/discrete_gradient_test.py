from unittest import TestCase

import numpy as np
from system.matrix.discrete_gradient import DiscreteGradient

from ...test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE


class TestLinearDiscreteGradient(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    discrete_gradient = DiscreteGradient(element_space)
    expected_discrete_gradient = (
        1 / 2 * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
    )

    def test_entries(self):
        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertAlmostEqual(
                    self.discrete_gradient[i, j], self.expected_discrete_gradient[i, j]
                )


class TestQuadraticDiscreteGradient(TestLinearDiscreteGradient):
    element_space = QUADRATIC_LAGRANGE_SPACE
    discrete_gradient = DiscreteGradient(element_space)
    expected_discrete_gradient = np.array(
        [
            [0, 2 / 3, 0, -2 / 3],
            [-2 / 3, 0, 2 / 3, 0],
            [0, -2 / 3, 0, 2 / 3],
            [2 / 3, 0, -2 / 3, 0],
        ]
    )
