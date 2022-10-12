from test.test_helper import VOLUME_MESH, VOLUME_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.system_vector import SWEGodunovNumericalFlux
from pde_solver.time_stepping import *


class GodunovTimeSteppingTest(TestCase):
    test_dofs = [
        np.array([[0.5, 0, 0.5, 0], [0.1, 0, 0.1, 0]]).T,
        np.array([[0, 0.5, 0, 0.5], [0, 0.2, 0, 0.2]]).T,
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
