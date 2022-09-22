from unittest import TestCase

import numpy as np
from system.matrix.artificial_diffusion import (
    BurgersArtificialDiffusion,
    DiscreteUpwind,
)
from system.matrix.discrete_gradient import DiscreteGradient
from system.matrix.mass import MassMatrix
from system.vector.group_finite_element_approximation import (
    GroupFiniteElementApproximation,
)
from system.vector.local_bounds import LocalMaximum, LocalMinimum
from system.vector.low_order_cg import LowOrderCGRightHandSide
from system.vector.lumped_mass import LumpedMassVector
from system.vector.mcl import MCLRightHandSide

from ...test_helper import (
    LINEAR_DOF_VECTOR,
    LINEAR_LAGRANGE_SPACE,
    QUADRATIC_DOF_VECTOR,
    QUADRATIC_LAGRANGE_SPACE,
)


class TestLinearLowOrderCGRightHandSide(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    mass = MassMatrix(LINEAR_LAGRANGE_SPACE)
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    artificial_diffusion = DiscreteUpwind(discrete_gradient)
    flux_approximation = GroupFiniteElementApproximation(dof_vector, lambda u: u)
    local_maximum = LocalMaximum(dof_vector)
    local_minimum = LocalMinimum(dof_vector)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [np.array([-4, 4, 0, 0])]
    mcl: MCLRightHandSide

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        low_cg = LowOrderCGRightHandSide(self.dof_vector)
        low_cg.lumped_mass = self.lumped_mass
        low_cg.discrete_gradient = self.discrete_gradient
        low_cg.flux_approximation = self.flux_approximation
        low_cg.artificial_diffusion = self.artificial_diffusion

        self.mcl = MCLRightHandSide(self.dof_vector)
        self.mcl.low_cg_right_hand_side = low_cg
        self.mcl.mass = self.mass
        self.mcl.local_maximum = self.local_maximum
        self.mcl.local_minimum = self.local_minimum

    def test_mcl(self):
        for dofs, expected_result in zip(
            self.test_dofs, self.expected_right_hand_sides
        ):
            self.dof_vector.dofs = dofs

            # for i in range(len(dofs)):
            #     self.assertAlmostEqual(
            #         self.mcl[i],
            #         expected_result[i],
            #         msg=f"index={i}, dofs={dofs}",
            #     )


# class TestQuadraticLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
#     dof_vector = QUADRATIC_DOF_VECTOR
#     lumped_mass = LumpedMassVector(QUADRATIC_LAGRANGE_SPACE)
#     discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
#     artificial_diffusion = DiscreteUpwind(discrete_gradient)
#     flux_approximation = GroupFiniteElementApproximation(dof_vector, lambda u: u)
#     test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
#     expected_right_hand_sides = [np.array([-8, 4, 0, 0]), np.array([24, -4, -8, -4])]


# class TestLinearBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
#     dof_vector = LINEAR_DOF_VECTOR
#     lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
#     discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
#     artificial_diffusion = BurgersArtificialDiffusion(dof_vector, discrete_gradient)
#     flux_approximation = GroupFiniteElementApproximation(
#         dof_vector, lambda u: 1 / 2 * u**2
#     )
#     test_dofs = [np.array([1, 2, 3, 4])]
#     expected_right_hand_sides = [np.array([40, -6, -10, -24])]


# class TestQuadraticBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
#     dof_vector = QUADRATIC_DOF_VECTOR
#     lumped_mass = LumpedMassVector(QUADRATIC_LAGRANGE_SPACE)
#     discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
#     artificial_diffusion = BurgersArtificialDiffusion(dof_vector, discrete_gradient)
#     flux_approximation = GroupFiniteElementApproximation(
#         dof_vector, lambda u: 1 / 2 * u**2
#     )
#     test_dofs = [np.array([1, 2, 3, 4])]
#     expected_right_hand_sides = [np.array([80, -6, -20, -24])]
