from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.discrete_solution import DiscreteSolution, DiscreteSolutionObservable
from pde_solver.system_matrix import (
    BurgersArtificialDiffusion,
    DiscreteGradient,
    DiscreteUpwind,
)
from pde_solver.system_vector import (
    FluxApproximation,
    LowOrderCGRightHandSide,
    LumpedMassVector,
)


class TestLinearLowOrderCGRightHandSide(TestCase):
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    artificial_diffusion = DiscreteUpwind(discrete_gradient)
    flux_approximation = FluxApproximation(lambda u: u)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [
        [-4, 4, 0, 0],
        [12, -4, -4, -4],
    ]
    right_hand_side: LowOrderCGRightHandSide

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        self.right_hand_side = LowOrderCGRightHandSide()
        self.right_hand_side.lumped_mass = self.lumped_mass
        self.right_hand_side.discrete_gradient = self.discrete_gradient
        self.right_hand_side.flux_approximation = self.flux_approximation
        self.right_hand_side.artificial_diffusion = self.artificial_diffusion

    def test_low_cg(self):
        for dofs, expected_result in zip(
            self.test_dofs, self.expected_right_hand_sides
        ):
            right_hand_side = self.right_hand_side(dofs)
            for i in range(len(right_hand_side)):
                self.assertAlmostEqual(
                    right_hand_side[i], expected_result[i], msg=f"index={i}"
                )


class TestQuadraticLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    lumped_mass = LumpedMassVector(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    artificial_diffusion = DiscreteUpwind(discrete_gradient)
    flux_approximation = FluxApproximation(lambda u: u)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[-8, 4, 0, 0], [24, -4, -8, -4]]


class TestLinearBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    raw_discrete_solution = DiscreteSolution(0, np.zeros(4))
    discrete_solution = DiscreteSolutionObservable(raw_discrete_solution)
    artificial_diffusion = BurgersArtificialDiffusion(
        discrete_gradient, discrete_solution
    )
    flux_approximation = FluxApproximation(lambda u: 1 / 2 * u**2)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[40, -6, -10, -24]]


class TestQuadraticBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    lumped_mass = LumpedMassVector(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    raw_discrete_solution = DiscreteSolution(0, np.zeros(4))
    discrete_solution = DiscreteSolutionObservable(raw_discrete_solution)
    artificial_diffusion = BurgersArtificialDiffusion(
        discrete_gradient, discrete_solution
    )
    flux_approximation = FluxApproximation(lambda u: 1 / 2 * u**2)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[80, -6, -20, -24]]
