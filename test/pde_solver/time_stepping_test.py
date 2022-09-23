from unittest import TestCase

import numpy as np
from system.matrix.artificial_diffusion import (
    BurgersArtificialDiffusion,
    DiscreteUpwind,
)
from system.matrix.discrete_gradient import DiscreteGradient
from system.vector.lumped_mass import LumpedMassVector

from pde_solver.time_stepping import *

from ..test_helper import LINEAR_DOF_VECTOR, LINEAR_LAGRANGE_SPACE, QUADRATIC_MESH


class TestSpatialMeshDependentTimeStepping(TestCase):
    time_stepping = SpatialMeshDependendetTimeStepping()
    time_stepping.start_time = 0
    time_stepping.end_time = 1
    time_stepping.mesh = QUADRATIC_MESH

    def test_uniform_time_stepping(self):
        self.time_stepping.cfl_number = 1
        expected_time_stepping = [0.5, 0.5]

        for i, delta_t in enumerate(self.time_stepping):
            expected_delta_t = expected_time_stepping[i]
            self.assertAlmostEqual(delta_t, expected_delta_t, msg=f"index={i}")

    def test_not_uniform_time_stepping(self):
        self.time_stepping.cfl_number = 0.75
        expected_time_stepping = [0.375, 0.375, 0.25]

        for i, delta_t in enumerate(self.time_stepping):
            expected_delta_t = expected_time_stepping[i]

            self.assertAlmostEqual(delta_t, expected_delta_t, msg=f"index={i}")


class TestAdaptiveMCLTimeStepping(TestCase):
    lumped_mass = LumpedMassVector(LINEAR_LAGRANGE_SPACE)
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)
    time_stepping = AdaptiveMCLTimeStepping()
    time_stepping.start_time = 0
    time_stepping.end_time = 1
    time_stepping.lumped_mass = lumped_mass

    def test_constant_diffusion(self):
        self.time_stepping.artificial_diffusion = DiscreteUpwind(self.discrete_gradient)
        self.time_stepping.cfl_number = 1
        expected_time_stepping = 4 * [0.25]

        for i, (delta_t, expected_delta_t) in enumerate(
            zip(self.time_stepping, expected_time_stepping)
        ):
            self.assertAlmostEqual(delta_t, expected_delta_t, msg=f"index={i}")

    def test_non_constant_diffusion(self):
        dof_vector = LINEAR_DOF_VECTOR
        self.time_stepping.artificial_diffusion = BurgersArtificialDiffusion(
            dof_vector, self.discrete_gradient
        )
        self.time_stepping.cfl_number = 1
        expected_time_stepping = [1 / 4, 1 / 8, 1 / 4, 1 / 8, 1 / 4]
        dofs = [np.array([1, 0, 1, 0]), np.array([2, 0, 2, 0])]

        dof_vector.dofs = dofs[0]

        for i, (delta_t, expected_delta_t) in enumerate(
            zip(self.time_stepping, expected_time_stepping)
        ):
            dof_vector.dofs = dofs[(i + 1) % 2]
            self.assertAlmostEqual(delta_t, expected_delta_t, msg=f"index={i}")
