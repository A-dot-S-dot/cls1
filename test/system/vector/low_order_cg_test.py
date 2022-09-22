from unittest import TestCase

import numpy as np
from system.matrix.artificial_diffusion import DiscreteUpwind
from system.matrix.discrete_gradient import DiscreteGradient
from system.vector.group_finite_element_approximation import (
    GroupFiniteElementApproximation,
)
from system.vector.low_order_cg import LowOrderCGRightHandSide
from system.vector.lumped_mass import LumpedMassVector

from ...test_helper import (
    LINEAR_DOF_VECTOR,
    LINEAR_LAGRANGE_SPACE,
    QUADRATIC_DOF_VECTOR,
    QUADRATIC_LAGRANGE_SPACE,
)


class TestLinearLowOrderCGRightHandSide(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    artificial_diffusion = DiscreteUpwind(discrete_gradient)
    flux_approximation = GroupFiniteElementApproximation(dof_vector, lambda u: u)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [
        np.array([-4, 4, 0, 0]),
        np.array([12, -4, -4, -4]),
    ]
    right_hand_side: LowOrderCGRightHandSide

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        self.right_hand_side = LowOrderCGRightHandSide(self.dof_vector)
        self.right_hand_side.lumped_mass = self.lumped_mass
        self.right_hand_side.discrete_gradient = self.discrete_gradient
        self.right_hand_side.flux_approximation = self.flux_approximation
        self.right_hand_side.artificial_diffusion = self.artificial_diffusion

    def test_advection_gradient(self):
        for dofs, expected_result in zip(
            self.test_dofs, self.expected_right_hand_sides
        ):
            self.dof_vector.dofs = dofs

            for i in range(len(dofs)):
                self.assertAlmostEqual(
                    self.right_hand_side[i],
                    expected_result[i],
                    msg=f"index={i}, dofs={dofs}",
                )


class TestQuadraticLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    dof_vector = QUADRATIC_DOF_VECTOR
    lumped_mass = LumpedMassVector(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    artificial_diffusion = DiscreteUpwind(discrete_gradient)
    flux_approximation = GroupFiniteElementApproximation(dof_vector, lambda u: u)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([-8, 4, 0, 0]), np.array([24, -4, -8, -4])]
