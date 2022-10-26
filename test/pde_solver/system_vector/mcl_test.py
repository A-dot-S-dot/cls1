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
    flux_approximation = vector.FluxApproximation(lambda u: u)
    low_cg = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    mcl = vector.MCLRightHandSide(element_space, low_cg, flux_approximation)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[12, -6, -4, -2]]

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

    def test_mcl(self):
        for dofs, expected_result in zip(
            self.test_dofs, self.expected_right_hand_sides
        ):
            mcl = self.mcl(dofs)
            for i in range(len(mcl)):
                self.assertAlmostEqual(mcl[i], expected_result[i], msg=f"index={i}")


class TestQuadraticLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = QUADRATIC_LAGRANGE_SPACE
    lumped_mass = vector.LumpedMassVector(element_space)
    artificial_diffusion = matrix.DiscreteUpwind(element_space)
    flux_gradient = vector.ApproximatedFluxGradient(element_space, lambda u: u)
    flux_approximation = vector.FluxApproximation(lambda u: u)
    low_cg = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    mcl = vector.MCLRightHandSide(element_space, low_cg, flux_approximation)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[24, -5.6, -9.6, -1.6]]


class TestLinearBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = LINEAR_LAGRANGE_SPACE
    lumped_mass = vector.LumpedMassVector(element_space)
    artificial_diffusion = matrix.BurgersArtificialDiffusion(element_space)
    flux_gradient = vector.ApproximatedFluxGradient(
        element_space, lambda u: 1 / 2 * u**2
    )
    flux_approximation = vector.FluxApproximation(lambda u: 1 / 2 * u**2)
    low_cg = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    mcl = vector.MCLRightHandSide(element_space, low_cg, flux_approximation)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[32, -37 / 3, -31 / 3, -28 / 3]]


class TestQuadraticBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = QUADRATIC_LAGRANGE_SPACE
    lumped_mass = vector.LumpedMassVector(element_space)
    artificial_diffusion = matrix.BurgersArtificialDiffusion(element_space)
    flux_gradient = vector.ApproximatedFluxGradient(
        element_space, lambda u: 1 / 2 * u**2
    )
    flux_approximation = vector.FluxApproximation(lambda u: 1 / 2 * u**2)
    low_cg = vector.LowOrderCGRightHandSide(
        lumped_mass, artificial_diffusion, flux_gradient
    )
    mcl = vector.MCLRightHandSide(element_space, low_cg, flux_approximation)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[64, -11.6, -26, -7.4]]
