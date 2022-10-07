from unittest import TestCase

import numpy as np
from pde_solver.system_matrix import MassMatrix, DiscreteGradient
from pde_solver.system_vector import (
    CGRightHandSide,
    AdvectionFluxGradient,
    ApproximatedFluxGradient,
)

from test.test_helper import (
    LINEAR_LAGRANGE_SPACE,
    QUADRATIC_LAGRANGE_SPACE,
)


class TestLinearCGRightHandSide(TestCase):
    mass = MassMatrix(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    flux_gradient = AdvectionFluxGradient(discrete_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([0, -3, 0, 3]), np.array([-6, 6, 6, -6])]
    right_hand_side: CGRightHandSide

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        self.right_hand_side = CGRightHandSide()
        self.right_hand_side.mass = self.mass
        self.right_hand_side.flux_gradient = self.flux_gradient

    def test_advection_gradient(self):
        for dofs, expected_result in zip(
            self.test_dofs, self.expected_right_hand_sides
        ):
            right_hand_side = self.right_hand_side(dofs)
            for i in range(len(right_hand_side)):
                self.assertAlmostEqual(
                    right_hand_side[i], expected_result[i], msg=f"index={i}"
                )


class TestQuadraticCGRightHandSide(TestLinearCGRightHandSide):
    mass = MassMatrix(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    flux_gradient = AdvectionFluxGradient(discrete_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([0, -2.5, 0, 2.5]), np.array([-8, 5, 8, -5])]


class TestLinearBurgersLowOrderCGRightHandSide(TestLinearCGRightHandSide):
    mass = MassMatrix(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    flux_gradient = ApproximatedFluxGradient(
        discrete_gradient, lambda u: 1 / 2 * u**2
    )
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [
        np.array([0, 1.5, 0, -1.5]),
        np.array([18, -12, -18, 12]),
    ]


class TestQuadraticBurgersLowOrderCGRightHandSide(TestLinearCGRightHandSide):
    mass = MassMatrix(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    flux_gradient = ApproximatedFluxGradient(
        discrete_gradient, lambda u: 1 / 2 * u**2
    )
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [
        np.array([0, 1.25, 0, -1.25]),
        np.array([24, -10, -24, 10]),
    ]
