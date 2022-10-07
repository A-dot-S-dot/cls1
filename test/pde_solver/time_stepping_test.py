from test.test_helper import (
    LINEAR_LAGRANGE_SPACE,
    QUADRATIC_MESH,
    VOLUME_MESH,
    VOLUME_SPACE,
)
from typing import List
from unittest import TestCase

import numpy as np

from pde_solver.system_matrix import (
    BurgersArtificialDiffusion,
    DiscreteGradient,
    DiscreteUpwind,
)
from pde_solver.system_vector import LumpedMassVector, SWEGodunovNumericalFlux
from pde_solver.time_stepping import *


class TestSpatialMeshDependentTimeStepping(TestCase):
    time_stepping = SpatialMeshDependendetTimeStepping()
    time_stepping.start_time = 0
    time_stepping.end_time = 1
    time_stepping.mesh = QUADRATIC_MESH

    def test_uniform_time_stepping(self):
        self._test_time_stepping(1, [0.5, 0.5])

    def _test_time_stepping(
        self, cfl_number: float, expected_time_stepping: List[float]
    ):
        self.time_stepping.cfl_number = cfl_number
        time_stepping = [self.time_stepping.time_step for _ in self.time_stepping]
        self.assertListEqual(time_stepping, expected_time_stepping)

    def test_not_uniform_time_stepping(self):
        self._test_time_stepping(0.75, [0.375, 0.375, 0.25])

    def test_to_small_break(self):
        self.assertRaises(TimeStepTooSmallError, self._test_time_stepping, 1e-13, [])


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


class GodunovTimeSteppingTest(TestCase):
    test_dofs = [
        np.array([[0.5, 0, 0.5, 0], [0.1, 0, 0.1, 0]]),
        np.array([[0, 0.5, 0, 0.5], [0, 0.2, 0, 0.2]]),
    ]
    numerical_flux = SWEGodunovNumericalFlux()
    numerical_flux.volume_space = VOLUME_SPACE
    numerical_flux.bottom_topography = np.array([1, 1, 1, 1])
    numerical_flux.gravitational_acceleration = 1
    time_stepping = SWEGodunovTimeStepping()
    time_stepping.start_time = 0
    time_stepping.end_time = 1
    time_stepping.mesh = VOLUME_MESH
    time_stepping.cfl_number = 1
    time_stepping.numerical_flux = numerical_flux
    expected_time_stepping = 10 * [
        0.13780075575721398,
        0.11290690484799541,
        0.13780075575721398,
        0.11290690484799541,
        0.13780075575721398,
        0.11290690484799541,
        0.13780075575721398,
        0.11007626242715796,
    ]

    def test_time_stepping(self):
        index = 0

        for (_, expected_time_step) in zip(
            self.time_stepping, self.expected_time_stepping
        ):
            self.numerical_flux(self.test_dofs[index % 2])
            time_step = self.time_stepping.time_step
            self.assertAlmostEqual(time_step, expected_time_step, msg=f"index={index}")

            index += 1
