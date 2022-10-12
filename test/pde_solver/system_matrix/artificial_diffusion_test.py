from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.discrete_solution import DiscreteSolution, DiscreteSolutionObservable
from pde_solver.system_matrix.artificial_diffusion import (
    BurgersArtificialDiffusion,
    DiscreteUpwind,
)
from pde_solver.system_matrix.discrete_gradient import DiscreteGradient


class TestLinearDiscreteUpwind(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    discrete_gradient = DiscreteGradient(element_space)
    discrete_upwind = DiscreteUpwind(discrete_gradient)
    expected_discrete_upwind = (
        1 / 2 * np.array([[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]])
    )

    def test_entries(self):
        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertAlmostEqual(
                    self.discrete_upwind[i, j], self.expected_discrete_upwind[i, j]
                )


class TestQuadraticDiscreteUpwind(TestLinearDiscreteUpwind):
    element_space = QUADRATIC_LAGRANGE_SPACE
    discrete_gradient = DiscreteGradient(element_space)
    discrete_upwind = DiscreteUpwind(discrete_gradient)
    expected_discrete_upwind = (
        1 / 3 * np.array([[-4, 2, 0, 2], [2, -4, 2, 0], [0, 2, -4, 2], [2, 0, 2, -4]])
    )


class TestLinearBurgersArtificialDiffusion(TestCase):
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    raw_discrete_solution = DiscreteSolution(0, np.zeros(4))
    discrete_solution = DiscreteSolutionObservable(raw_discrete_solution)
    burgers_diffusion = BurgersArtificialDiffusion(discrete_gradient, discrete_solution)
    test_dofs = np.array([1, 2, 3, 4])
    expected_burgers_diffusion = (
        1 / 2 * np.array([[-6, 2, 0, 4], [2, -5, 3, 0], [0, 3, -7, 4], [4, 0, 4, -8]])
    )

    def test_entries(self):
        self.burgers_diffusion.assemble(self.test_dofs)

        for i in range(self.discrete_gradient.dimension):
            for j in range(self.discrete_gradient.dimension):
                self.assertAlmostEqual(
                    self.burgers_diffusion[i, j], self.expected_burgers_diffusion[i, j]
                )


class TestQuadraticBurgersArtificialDiffusion(TestLinearBurgersArtificialDiffusion):
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    raw_discrete_solution = DiscreteSolution(0, np.zeros(4))
    discrete_solution = DiscreteSolutionObservable(raw_discrete_solution)
    burgers_diffusion = BurgersArtificialDiffusion(discrete_gradient, discrete_solution)
    expected_burgers_diffusion = (
        2 / 3 * np.array([[-6, 2, 0, 4], [2, -5, 3, 0], [0, 3, -7, 4], [4, 0, 4, -8]])
    )
