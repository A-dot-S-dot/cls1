from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np
import pde_solver.system_matrix as matrix
import pde_solver.system_vector as vector


class TestLinearLowOrderCGRightHandSide(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    lumped_mass = vector.LumpedMassVector(element_space)
    artificial_diffusion = matrix.DiscreteUpwind(element_space)
    flux_gradient = vector.ApproximatedFluxGradient(element_space, lambda u: u)
    right_hand_side = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [
        [-4, 4, 0, 0],
        [12, -4, -4, -4],
    ]

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
    element_space = QUADRATIC_LAGRANGE_SPACE
    lumped_mass = vector.LumpedMassVector(element_space)
    artificial_diffusion = matrix.DiscreteUpwind(element_space)
    flux_gradient = vector.ApproximatedFluxGradient(element_space, lambda u: u)
    right_hand_side = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[-8, 4, 0, 0], [24, -4, -8, -4]]


class TestLinearBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = LINEAR_LAGRANGE_SPACE
    lumped_mass = vector.LumpedMassVector(element_space)
    artificial_diffusion = matrix.BurgersArtificialDiffusion(element_space)
    flux_gradient = vector.ApproximatedFluxGradient(
        element_space, lambda u: 1 / 2 * u**2
    )
    right_hand_side = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[40, -6, -10, -24]]


class TestQuadraticBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = QUADRATIC_LAGRANGE_SPACE
    lumped_mass = vector.LumpedMassVector(element_space)
    artificial_diffusion = matrix.BurgersArtificialDiffusion(element_space)
    flux_gradient = vector.ApproximatedFluxGradient(
        element_space, lambda u: 1 / 2 * u**2
    )
    right_hand_side = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[80, -6, -20, -24]]
