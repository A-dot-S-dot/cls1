from unittest import TestCase

import numpy as np
from system.matrix.discrete_gradient import DiscreteGradient
from system.matrix.mass import MassMatrix
from system.vector.cg import CGRightHandSide
from system.vector.flux_gradient import AdvectionFluxGradient

from ...test_helper import (
    LINEAR_DOF_VECTOR,
    LINEAR_LAGRANGE_SPACE,
    QUADRATIC_DOF_VECTOR,
    QUADRATIC_LAGRANGE_SPACE,
)


class TestLinearCGRightHandSide(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    mass = MassMatrix(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    flux_gradient = AdvectionFluxGradient(dof_vector, discrete_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([0, -3, 0, 3]), np.array([-6, 6, 6, -6])]
    right_hand_side: CGRightHandSide

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        self.right_hand_side = CGRightHandSide(self.dof_vector)
        self.right_hand_side.mass = self.mass
        self.right_hand_side.flux_gradient = self.flux_gradient

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


class TestQuadraticCGRightHandSide(TestLinearCGRightHandSide):
    dof_vector = QUADRATIC_DOF_VECTOR
    mass = MassMatrix(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    flux_gradient = AdvectionFluxGradient(dof_vector, discrete_gradient)
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([0, -2.5, 0, 2.5]), np.array([-8, 5, 8, -5])]
