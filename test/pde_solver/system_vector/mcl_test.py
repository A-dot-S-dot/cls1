from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.discrete_solution import DiscreteSolution, DiscreteSolutionObservable
from pde_solver.system_matrix import (
    BurgersArtificialDiffusion,
    DiscreteGradient,
    DiscreteUpwind,
    MassMatrix,
)
from pde_solver.system_vector import (
    FluxApproximation,
    LocalMaximum,
    LocalMinimum,
    LowOrderCGRightHandSide,
    LumpedMassVector,
    MCLRightHandSide,
)


class TestLinearLowOrderCGRightHandSide(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    mass = MassMatrix(LINEAR_LAGRANGE_SPACE)
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    artificial_diffusion = DiscreteUpwind(discrete_gradient)
    flux_approximation = FluxApproximation(lambda u: u)
    local_maximum = LocalMaximum(LINEAR_LAGRANGE_SPACE)
    local_minimum = LocalMinimum(LINEAR_LAGRANGE_SPACE)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[12, -6, -4, -2]]
    mcl: MCLRightHandSide

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        low_cg = LowOrderCGRightHandSide()
        low_cg.lumped_mass = self.lumped_mass
        low_cg.discrete_gradient = self.discrete_gradient
        low_cg.flux_approximation = self.flux_approximation
        low_cg.artificial_diffusion = self.artificial_diffusion

        self.mcl = MCLRightHandSide(self.element_space)
        self.mcl.low_cg_right_hand_side = low_cg
        self.mcl.mass = self.mass
        self.mcl.local_maximum = self.local_maximum
        self.mcl.local_minimum = self.local_minimum

    def test_mcl(self):
        for dofs, expected_result in zip(
            self.test_dofs, self.expected_right_hand_sides
        ):
            mcl = self.mcl(dofs)
            for i in range(len(mcl)):
                self.assertAlmostEqual(mcl[i], expected_result[i], msg=f"index={i}")


class TestQuadraticLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = QUADRATIC_LAGRANGE_SPACE
    mass = MassMatrix(QUADRATIC_LAGRANGE_SPACE)
    lumped_mass = LumpedMassVector(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    artificial_diffusion = DiscreteUpwind(discrete_gradient)
    flux_approximation = FluxApproximation(lambda u: u)
    local_maximum = LocalMaximum(QUADRATIC_LAGRANGE_SPACE)
    local_minimum = LocalMinimum(QUADRATIC_LAGRANGE_SPACE)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[24, -5.6, -9.6, -1.6]]


class TestLinearBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = LINEAR_LAGRANGE_SPACE
    mass = MassMatrix(LINEAR_LAGRANGE_SPACE)
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    raw_discrete_solution = DiscreteSolution(0, np.zeros(4))
    discrete_solution = DiscreteSolutionObservable(raw_discrete_solution)
    artificial_diffusion = BurgersArtificialDiffusion(
        discrete_gradient, discrete_solution
    )
    flux_approximation = FluxApproximation(lambda u: 1 / 2 * u**2)
    local_maximum = LocalMaximum(LINEAR_LAGRANGE_SPACE)
    local_minimum = LocalMinimum(LINEAR_LAGRANGE_SPACE)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[32, -37 / 3, -31 / 3, -28 / 3]]


class TestQuadraticBurgersLowOrderCGRightHandSide(TestLinearLowOrderCGRightHandSide):
    element_space = QUADRATIC_LAGRANGE_SPACE
    mass = MassMatrix(QUADRATIC_LAGRANGE_SPACE)
    lumped_mass = LumpedMassVector(QUADRATIC_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(QUADRATIC_LAGRANGE_SPACE)
    raw_discrete_solution = DiscreteSolution(0, np.zeros(4))
    discrete_solution = DiscreteSolutionObservable(raw_discrete_solution)
    artificial_diffusion = BurgersArtificialDiffusion(
        discrete_gradient, discrete_solution
    )
    flux_approximation = FluxApproximation(lambda u: 1 / 2 * u**2)
    local_maximum = LocalMaximum(QUADRATIC_LAGRANGE_SPACE)
    local_minimum = LocalMinimum(QUADRATIC_LAGRANGE_SPACE)
    test_dofs = [np.array([1, 2, 3, 4])]
    expected_right_hand_sides = [[64, -11.6, -26, -7.4]]
