from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np
from lib import (
    AdvectionFluxGradient,
    ApproximatedFluxGradient,
    DiscreteGradient,
    MassMatrix,
)
from scalar.solver.cg import CGRightHandSide


class TestLinearCGRightHandSide(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    mass = MassMatrix(element_space)
    flux_gradient = AdvectionFluxGradient(element_space)
    right_hand_side = CGRightHandSide(mass, flux_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([0, 3, 0, -3]), np.array([6, -6, -6, 6])]

    def test_cg(self):
        for dofs, expected_result in zip(
            self.test_dofs, self.expected_right_hand_sides
        ):
            right_hand_side = self.right_hand_side(0.0, dofs)
            for i in range(len(right_hand_side)):
                self.assertAlmostEqual(
                    right_hand_side[i], expected_result[i], msg=f"index={i}"
                )


class TestQuadraticCGRightHandSide(TestLinearCGRightHandSide):
    element_space = QUADRATIC_LAGRANGE_SPACE
    mass = MassMatrix(element_space)
    flux_gradient = AdvectionFluxGradient(element_space)
    right_hand_side = CGRightHandSide(mass, flux_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([0, 2.5, 0, -2.5]), np.array([8, -5, -8, 5])]


class TestLinearBurgersLowOrderCGRightHandSide(TestLinearCGRightHandSide):
    element_space = LINEAR_LAGRANGE_SPACE
    mass = MassMatrix(element_space)
    flux_gradient = ApproximatedFluxGradient(element_space, lambda u: 1 / 2 * u**2)
    right_hand_side = CGRightHandSide(mass, flux_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [
        np.array([0, 1.5, 0, -1.5]),
        np.array([18, -12, -18, 12]),
    ]


class TestQuadraticBurgersLowOrderCGRightHandSide(TestLinearCGRightHandSide):
    element_space = QUADRATIC_LAGRANGE_SPACE
    mass = MassMatrix(element_space)
    discrete_gradient = DiscreteGradient(element_space)
    flux_gradient = ApproximatedFluxGradient(element_space, lambda u: 1 / 2 * u**2)
    right_hand_side = CGRightHandSide(mass, flux_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [
        np.array([0, 1.25, 0, -1.25]),
        np.array([24, -10, -24, 10]),
    ]
