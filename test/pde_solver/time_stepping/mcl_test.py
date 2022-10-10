from test.test_helper import (
    LINEAR_LAGRANGE_SPACE,
)
from unittest import TestCase

import numpy as np

from pde_solver.system_matrix import (
    BurgersArtificialDiffusion,
    DiscreteGradient,
    DiscreteUpwind,
)
from pde_solver.system_vector import LumpedMassVector
from pde_solver.time_stepping import *


class TestMCLTimeStepping(TestCase):
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    discrete_solution = DiscreteSolutionObservable(0, np.zeros(4))
    adaptive_time_stepping = AdaptiveMCLTimeStepping(discrete_solution)
    adaptive_time_stepping.start_time = 0
    adaptive_time_stepping.end_time = 1
    adaptive_time_stepping.lumped_mass = lumped_mass
    constant_time_stepping = MCLTimeStepping()
    constant_time_stepping.start_time = 0
    constant_time_stepping.end_time = 1
    constant_time_stepping.lumped_mass = lumped_mass

    def test_constant_diffusion(self):
        self.constant_time_stepping.artificial_diffusion = DiscreteUpwind(
            self.discrete_gradient
        )
        self.constant_time_stepping.cfl_number = 1
        self.constant_time_stepping.update_time_step()
        expected_time_stepping = 4 * [0.125]

        for i, (_, expected_delta_t) in enumerate(
            zip(self.constant_time_stepping, expected_time_stepping)
        ):
            time_step = self.constant_time_stepping.time_step
            self.assertAlmostEqual(time_step, expected_delta_t, msg=f"index={i}")

    def test_non_constant_diffusion(self):
        self.adaptive_time_stepping.artificial_diffusion = BurgersArtificialDiffusion(
            self.discrete_gradient, self.discrete_solution
        )
        self.adaptive_time_stepping.cfl_number = 1
        expected_time_stepping = [1 / 4, 1 / 8, 1 / 4, 1 / 8, 1 / 4]
        dofs = [np.array([0.5, 0, 0.5, 0]), np.array([1, 0, 1, 0])]

        self.adaptive_time_stepping.artificial_diffusion.assemble(dofs[0])
        self.adaptive_time_stepping.update_time_step()

        for i, (_, expected_delta_t) in enumerate(
            zip(self.adaptive_time_stepping, expected_time_stepping)
        ):
            time_step = self.adaptive_time_stepping.time_step
            self.assertAlmostEqual(time_step, expected_delta_t, msg=f"index={i}")

            self.adaptive_time_stepping.artificial_diffusion.assemble(dofs[(i + 1) % 2])
            self.adaptive_time_stepping.update_time_step()

    def test_satisfy_cfl(self):
        self.constant_time_stepping.artificial_diffusion = BurgersArtificialDiffusion(
            self.discrete_gradient, self.discrete_solution
        )
        self.constant_time_stepping.cfl_number = 1
        self.constant_time_stepping.artificial_diffusion.assemble(
            np.array([1, 0, 1, 0])
        )
        self.constant_time_stepping.update_time_step()

        self.assertTrue(self.constant_time_stepping.satisfy_cfl())

        self.constant_time_stepping.artificial_diffusion.assemble(
            np.array([2, 0, 2, 0])
        )
        self.assertFalse(self.constant_time_stepping.satisfy_cfl())
